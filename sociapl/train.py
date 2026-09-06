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
    p.add_argument("--fresh", type=int, default=0, help="1 = ignore an existing <out>/ckpt.pt and start over")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--wandb", action="store_true", help="mirror log.csv to Weights & Biases")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_name", default=None, help="defaults to the <out> basename")
    p.add_argument("--wandb_tags", default="", help="comma-separated")
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), **HP}, open(f"{a.out}/config.json", "w"), indent=2)

    workers = [Worker(a.mode, a.n_experts, a.n_goals, a.expert_eps, a.p_social, seed=a.seed * 1000 + i) for i in range(a.n_envs)]
    net = SociAPLNet(aux=a.aux)
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"])
    print(f"params: {sum(p.numel() for p in net.parameters()):,}", flush=True)

    # --- checkpoint restore (mirrors train_ethics.py) -----------------------
    # Resume matters even here: a SLURM pass that runs out of walltime must be
    # able to continue rather than silently restart from scratch and truncate
    # its own log.
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    total = 0
    resume_from = None
    if not a.fresh and os.path.exists(ckpt_path):
        resume_from = ckpt_path
    elif a.init:
        resume_from = a.init
    if resume_from:
        st = torch.load(resume_from, map_location="cpu", weights_only=True)
        if isinstance(st, dict) and "net" in st:
            net.load_state_dict(st["net"])
            if "opt" in st and resume_from == ckpt_path:
                opt.load_state_dict(st["opt"])
            if resume_from == ckpt_path:
                total = int(st.get("episodes", 0))
        else:  # old weights-only checkpoint
            net.load_state_dict(st)
        print(f"resumed from {resume_from} at episode {total}", flush=True)
    if total >= a.episodes:
        print("target episode count already reached; nothing to do", flush=True)
        return

    run = None
    if a.wandb:
        import wandb_utils
        base = os.path.basename(a.out.rstrip("/"))
        run = wandb_utils.init(
            a.out, config={**vars(a), **HP},
            project=a.wandb_project, entity=a.wandb_entity,
            group=a.wandb_group or base.rsplit("_s", 1)[0],
            name=a.wandb_name or base, job_type="train",
            tags=[a.mode, f"aux_{a.aux}", *a.wandb_tags.split(",")],
        )

    def save_ckpt():
        tmp = ckpt_path + ".tmp"
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "episodes": total}, tmp)
        os.replace(tmp, ckpt_path)  # atomic on POSIX

    obs = np.stack([w.reset() for w in workers]); h, c = net.init_state(a.n_envs)
    log_path = os.path.join(a.out, "log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="", buffering=1); log = csv.writer(logf)
    HEADER = ["episodes", "learner_return", "expert_return", "frac_social",
              "l_pi", "l_v", "l_aux", "ent", "kl", "sec"]
    if new_log:
        log.writerow(HEADER)
    t0 = time.time()
    while total < a.episodes:
        stats = []
        data, obs, h, c = collect(net, workers, a.batch_episodes, obs, h, c, stats)
        info = update(net, opt, data, a.aux)
        total += len(stats)
        lr_ = np.mean([s["ep_return"] for s in stats]); er = np.mean([s["ep_expert_return"] for s in stats])
        fs = np.mean([s["social"] for s in stats])
        row = [total, lr_, er, fs, info.get("pi"), info.get("v"), info.get("aux"), info.get("ent"), info.get("kl"), time.time() - t0]
        log.writerow(row); logf.flush()
        if run is not None:
            wandb_utils.log_row(run, HEADER, row)
        print(" ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in row), flush=True)
        save_ckpt()

    if run is not None:
        wandb_utils.summarize(run, log_path)
        wandb_utils.finish(run)


if __name__ == "__main__":
    main()
