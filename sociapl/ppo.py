"""Recurrent PPO with GAE and the SociAPL auxiliary loss (Appendix 7.8 hyperparameters).

Deviation from the paper (documented): we do not recompute stored hidden states and
advantages every 2 minibatches; segments start from the hidden state stored at rollout time.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from model import N_ACTIONS

HP = dict(lr=1e-4, gamma=0.993, lam=0.97, clip=0.2, kl_target=0.01, kl_hard=0.03,
          c_v=0.1, c_aux=3.0, c_ent=1e-5, minibatches=20, mb_trajs=512, seg_len=16)


class Rollout:
    def __init__(self):
        self.obs, self.next_obs, self.act, self.logp, self.val, self.rew, self.mask = ([] for _ in range(7))
        self.h, self.c = [], []

    def add(self, obs, next_obs, act, logp, val, rew, mask, h, c):
        self.obs.append(obs); self.next_obs.append(next_obs); self.act.append(act); self.logp.append(logp)
        self.val.append(val); self.rew.append(rew); self.mask.append(mask); self.h.append(h); self.c.append(c)

    def stack(self):
        f = lambda xs: torch.stack(xs)  # (T, B, ...)
        return dict(obs=f(self.obs), next_obs=f(self.next_obs), act=f(self.act), logp=f(self.logp),
                    val=f(self.val), rew=f(self.rew), mask=f(self.mask), h=f(self.h), c=f(self.c))


def collect(net, workers, n_episodes, obs, h, c, stats):
    """Run workers until n_episodes episodes complete in total. obs/h/c carry across calls."""
    B = len(workers)
    ro = Rollout()
    mask = torch.ones(B)
    done_eps = 0
    while done_eps < n_episodes:
        obs_t = torch.as_tensor(obs)
        a, logp, v, h_new, c_new = net.act(obs_t, h, c)
        next_obs = np.empty_like(obs); rew = np.zeros(B, np.float32); new_mask = torch.ones(B)
        for i, w in enumerate(workers):
            o, r, d, info = w.step(a[i].item())
            rew[i] = r
            if d:
                stats.append(info); done_eps += 1
                new_mask[i] = 0.0
                o = w.reset()
            next_obs[i] = o
        # next_obs stored for aux target is the true next frame (pre-reset), obs at t+1 is post-reset
        true_next = np.stack([workers[i].env.gen_obs()[0] if new_mask[i] > 0 else next_obs[i] for i in range(B)]).astype(np.uint8)
        ro.add(obs_t, torch.as_tensor(true_next), a, logp, v, torch.as_tensor(rew), mask, h[0], c[0])
        obs, h, c, mask = next_obs, h_new, c_new, new_mask
    with torch.no_grad():
        _, _, last_v, _, _ = net.act(torch.as_tensor(obs), h, c)
    data = ro.stack()
    data["last_val"] = last_v; data["last_mask"] = mask
    return data, obs, h, c


def gae(data, gamma, lam):
    T, B = data["rew"].shape
    adv = torch.zeros(T, B); last = torch.zeros(B)
    next_v = data["last_val"]; next_m = data["last_mask"]
    for t in reversed(range(T)):
        delta = data["rew"][t] + gamma * next_v * next_m - data["val"][t]
        last = delta + gamma * lam * next_m * last
        adv[t] = last
        next_v, next_m = data["val"][t], data["mask"][t]
    return adv, adv + data["val"]


def update(net, opt, data, aux_mode, hp=HP):
    T, B = data["rew"].shape
    adv, ret = gae(data, hp["gamma"], hp["lam"])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    L = hp["seg_len"]
    starts = [(t, b) for b in range(B) for t in range(0, T - L + 1, L)]
    snapshot = copy.deepcopy(net.state_dict()); opt_snap = copy.deepcopy(opt.state_dict())
    logs = {"pi": [], "v": [], "aux": [], "ent": [], "kl": []}
    for it in range(hp["minibatches"]):
        idx = np.random.choice(len(starts), min(hp["mb_trajs"], len(starts)), replace=False)
        sel = [starts[i] for i in idx]
        g = lambda k: torch.stack([data[k][t:t + L, b] for t, b in sel], 1)  # (L, N, ...)
        obs, nobs, act, old_logp, mask = g("obs"), g("next_obs"), g("act"), g("logp"), g("mask")
        A, R = g_adv(adv, sel, L), g_adv(ret, sel, L)
        h0 = torch.stack([data["h"][t, b] for t, b in sel], 0).unsqueeze(0)
        c0 = torch.stack([data["c"][t, b] for t, b in sel], 0).unsqueeze(0)
        feat, _ = net.forward_seq(obs, h0, c0, mask)
        logits, value = net.heads(feat)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act)
        ratio = torch.exp(logp - old_logp)
        l_pi = -torch.min(ratio * A, torch.clamp(ratio, 1 - hp["clip"], 1 + hp["clip"]) * A).mean()
        l_v = F.mse_loss(value, R)
        ent = dist.entropy().mean()
        if aux_mode == "none":
            l_aux = torch.zeros(())
        else:
            target = (nobs if aux_mode == "pred" else obs).float() / 255.0
            l_aux = (net.aux_predict(feat, act) - target).abs().mean()
        loss = l_pi + hp["c_v"] * l_v + hp["c_aux"] * l_aux - hp["c_ent"] * ent
        kl = (old_logp - logp).mean().item()
        if kl > hp["kl_hard"]:
            net.load_state_dict(snapshot); opt.load_state_dict(opt_snap)
            logs["reverted"] = True
            break
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        logs["pi"].append(l_pi.item()); logs["v"].append(l_v.item()); logs["aux"].append(l_aux.item())
        logs["ent"].append(ent.item()); logs["kl"].append(kl)
        if kl > hp["kl_target"]:
            break
    return {k: (float(np.mean(v)) if isinstance(v, list) and v else v) for k, v in logs.items()}


def g_adv(x, sel, L):
    return torch.stack([x[t:t + L, b] for t, b in sel], 1)
