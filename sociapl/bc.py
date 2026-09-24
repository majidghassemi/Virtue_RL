"""Behaviour-cloning baseline (Appendix 7.4): same encoder+LSTM, trained on the expert's own POV.
Requires privileged access to expert observations and actions, which the social learner never gets.

  python bc.py --episodes 2000 --epochs 10 --out runs/bc_s0
"""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
from envs import Worker
from model import SociAPLNet, get_device


def collect(n_episodes, seed):
    w = Worker("social", n_experts=2, n_goals=3, seed=seed)
    O, A = [], []
    for _ in range(n_episodes):
        w.reset(); done = False; o_ep, a_ep = [], []
        while not done:
            o, a_exp = w.expert_obs_and_actions()
            o_ep.append(o); a_ep.append(a_exp)
            _, _, done, _ = w.step(np.random.randint(3))  # learner slot acts randomly; irrelevant to expert data
        O.append(np.stack(o_ep)); A.append(np.array(a_ep))
    return np.stack(O), np.stack(A)  # (N, T, 21,21,3), (N, T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="runs/bc")
    p.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps or cpu")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
    dev = get_device(a.device); print(f"device: {dev}", flush=True)
    O, A = collect(a.episodes, a.seed)
    O = torch.as_tensor(O).permute(1, 0, 2, 3, 4); A = torch.as_tensor(A).permute(1, 0)  # (T,N,...)
    net = SociAPLNet(aux="none").to(dev); opt = torch.optim.Adam(net.parameters(), 1e-4)
    T, N = A.shape; bs = 32
    for ep in range(a.epochs):
        perm = torch.randperm(N); tot = 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            h, c = net.init_state(len(idx))
            # dataset stays in host memory; only the minibatch is moved to the device
            o, y = O[:, idx].to(dev, non_blocking=True), A[:, idx].to(dev, non_blocking=True)
            feat, _ = net.forward_seq(o, h, c, torch.ones(T, len(idx), device=dev))
            logits, _ = net.heads(feat)
            loss = F.cross_entropy(logits.reshape(-1, 7), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"epoch {ep} loss {tot / max(1, N // bs):.4f}", flush=True)
    torch.save(net.state_dict(), f"{a.out}/ckpt.pt")


if __name__ == "__main__":
    main()
