"""SociAPL network: conv encoder -> FC -> LSTM -> {policy, value, auxiliary} heads.

Layer sizes follow Appendix 7.7 of Ndousse et al. (2021). ~670k parameters.
aux in {'pred', 'rec', 'none'}: next-state prediction, current-state reconstruction, or no aux head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 7


def get_device(name="auto"):
    """'auto' -> cuda if available, else mps (Apple GPU), else cpu. Otherwise passed to torch.device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_weights(path, device="cpu"):
    """Network state_dict from either a weights-only file (train.py, bc.py) or a
    train_ethics.py checkpoint {net, opt, episodes}, mapped onto `device`."""
    st = torch.load(path, map_location=device)
    return st["net"] if isinstance(st, dict) and "net" in st else st


class SociAPLNet(nn.Module):
    def __init__(self, aux="pred", hidden=192):
        super().__init__()
        self.aux = aux
        self.hidden = hidden
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=3), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3), nn.LeakyReLU(),
        )  # 21x21 -> 7x7 -> 5x5 -> 3x3, 64*9 = 576
        self.fc = nn.Sequential(nn.Linear(576, hidden), nn.Tanh())
        self.lstm = nn.LSTM(hidden, hidden)
        self.pi = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, N_ACTIONS))
        self.v = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        if aux != "none":
            self.aux_fc = nn.Sequential(nn.Linear(hidden + N_ACTIONS, 576), nn.Tanh())
            self.aux_deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 3), nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, 3), nn.LeakyReLU(),
                nn.ConvTranspose2d(32, 3, 3, stride=3),
            )  # 3x3 -> 5x5 -> 7x7 -> 21x21

    def encode(self, obs):
        # obs: (..., 21, 21, 3) uint8 -> float in [0,1], channels first
        shp = obs.shape[:-3]
        x = obs.reshape(-1, 21, 21, 3).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        z = self.fc(self.conv(x).flatten(1))
        return z.reshape(*shp, self.hidden)

    def forward_seq(self, obs, h0, c0, masks):
        """obs: (T, B, 21,21,3); h0,c0: (1,B,H); masks: (T,B) 1.0 = continue, 0.0 = new episode at t.
        Runs the LSTM step by step so hidden state resets at episode boundaries."""
        T, B = obs.shape[:2]
        z = self.encode(obs)  # (T,B,H)
        h, c = h0, c0
        outs = []
        for t in range(T):
            m = masks[t].view(1, B, 1)
            h, c = h * m, c * m
            out, (h, c) = self.lstm(z[t:t + 1], (h, c))
            outs.append(out)
        feat = torch.cat(outs, 0)  # (T,B,H)
        return feat, (h, c)

    def heads(self, feat):
        return self.pi(feat), self.v(feat).squeeze(-1)

    def aux_predict(self, feat, actions):
        """Predict next (or current) observation in [0,1], channels-last, from LSTM features + action."""
        a1h = F.one_hot(actions, N_ACTIONS).float()
        x = self.aux_fc(torch.cat([feat, a1h], -1))
        shp = x.shape[:-1]
        x = x.reshape(-1, 64, 3, 3)
        img = self.aux_deconv(x)  # (N,3,21,21)
        img = torch.sigmoid(img).permute(0, 2, 3, 1)
        return img.reshape(*shp, 21, 21, 3)

    @torch.no_grad()
    def act(self, obs, h, c, deterministic=False):
        """Single step for B parallel envs. obs: (B,21,21,3)."""
        feat, (h, c) = self.forward_seq(obs.unsqueeze(0), h, c, torch.ones(1, obs.shape[0], device=obs.device))
        logits, value = self.heads(feat[0])
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a, dist.log_prob(a), value, h, c

    def init_state(self, B):
        dev = next(self.parameters()).device
        return torch.zeros(1, B, self.hidden, device=dev), torch.zeros(1, B, self.hidden, device=dev)
