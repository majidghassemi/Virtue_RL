"""Evaluate a checkpoint on the paper's H1/H2 conditions.

  python evaluate.py --ckpt runs/social_pred_s0/ckpt.pt --aux pred --episodes 50
Reports mean learner return in: solo 3-goal, social 3-goal, social 4-goal (zero-shot transfer), solo 4-goal.
"""
import argparse, json
import numpy as np, torch
from envs import Worker
from model import SociAPLNet


def run(net, mode, n_goals, episodes, seed=123, n_experts=2, expert_eps=0.0):
    w = Worker(mode, n_experts, n_goals, expert_eps, seed=seed)
    rets, erets = [], []
    for ep in range(episodes):
        obs = w.reset(); h, c = net.init_state(1); done = False
        while not done:
            a, _, _, h, c = net.act(torch.as_tensor(obs[None]), h, c)
            obs, r, done, info = w.step(a.item())
        rets.append(info["ep_return"]); erets.append(info["ep_expert_return"])
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(erets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--aux", default="pred")
    p.add_argument("--episodes", type=int, default=50); p.add_argument("--out", default=None)
    a = p.parse_args()
    net = SociAPLNet(aux=a.aux); net.load_state_dict(torch.load(a.ckpt)); net.eval()
    res = {}
    for name, mode, ng in [("solo_3goal", "solo", 3), ("social_3goal", "social", 3),
                           ("social_4goal", "social", 4), ("solo_4goal", "solo", 4)]:
        m, s, e = run(net, mode, ng, a.episodes)
        res[name] = dict(learner_mean=m, learner_std=s, expert_mean=e)
        print(f"{name:14s} learner {m:6.2f} ± {s:4.2f}   expert {e:6.2f}")
    if a.out: json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
