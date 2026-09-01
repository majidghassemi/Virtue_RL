"""Train a SociAPL learner.

Examples (paper-scale is --episodes 1500000 --batch_episodes 128):
  python train.py --mode social --aux pred  --seed 0 --out runs/social_pred_s0
  python train.py --mode solo   --aux pred  --seed 0 --out runs/solo_pred_s0
  python train.py --mode social --aux none  --seed 0 --out runs/social_vanilla_s0
  python train.py --mode social --aux rec   --seed 0 --out runs/social_rec_s0
  python train.py --mode mixed  --aux pred  --seed 0 --out runs/mixed_pred_s0 --init runs/social_pred_s0/ckpt.pt
"""
import argparse, csv, json, os, time
import numpy as np, torch
from envs import Worker
from model import SociAPLNet
from ppo import collect, update, HP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["solo", "social", "mixed"], default="social")
    p.add_argument("--aux", choices=["pred", "rec", "none"], default="pred")
    p.add_argument("--episodes", type=int, default=1_500_000)
    p.add_argument("--batch_episodes", type=int, default=128)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_goals", type=int, default=3)
    p.add_argument("--n_experts", type=int, default=2)
    p.add_argument("--expert_eps", type=float, default=0.0, help=">0 for imperfect experts")
    p.add_argument("--p_social", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/debug")
    p.add_argument("--init", default=None, help="checkpoint to initialise from (for mixed-schedule continuation)")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), **HP}, open(f"{a.out}/config.json", "w"), indent=2)

    workers = [Worker(a.mode, a.n_experts, a.n_goals, a.expert_eps, a.p_social, seed=a.seed * 1000 + i) for i in range(a.n_envs)]
    net = SociAPLNet(aux=a.aux)
    if a.init:
        net.load_state_dict(torch.load(a.init))
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"])
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")

    obs = np.stack([w.reset() for w in workers]); h, c = net.init_state(a.n_envs)
    logf = open(f"{a.out}/log.csv", "w", newline="", buffering=1); log = csv.writer(logf); log.writerow(
        ["episodes", "learner_return", "expert_return", "frac_social", "l_pi", "l_v", "l_aux", "ent", "kl", "sec"])
    total, t0 = 0, time.time()
    while total < a.episodes:
        stats = []
        data, obs, h, c = collect(net, workers, a.batch_episodes, obs, h, c, stats)
        info = update(net, opt, data, a.aux)
        total += len(stats)
        lr_ = np.mean([s["ep_return"] for s in stats]); er = np.mean([s["ep_expert_return"] for s in stats])
        fs = np.mean([s["social"] for s in stats])
        row = [total, lr_, er, fs, info.get("pi"), info.get("v"), info.get("aux"), info.get("ent"), info.get("kl"), time.time() - t0]
        log.writerow(row); logf.flush(); print(" ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in row), flush=True)
        torch.save(net.state_dict(), f"{a.out}/ckpt.pt")


if __name__ == "__main__":
    main()
