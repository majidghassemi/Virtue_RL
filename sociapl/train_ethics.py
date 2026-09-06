"""Train on Ethical Goal Cycle. The experiment grid:

  R0 main:      python train_ethics.py --mode social --virtuous 1 --harm_delivery none    --out runs/e_r0_virt_sS --seed S
  R0 control:   python train_ethics.py --mode social --virtuous 0 --harm_delivery none    --out runs/e_r0_short_sS --seed S
  R0 solo:      python train_ethics.py --mode solo                 --harm_delivery none    --out runs/e_r0_solo_sS --seed S
  R2 baseline:  python train_ethics.py --mode solo                 --harm_delivery dense   --out runs/e_r2_solo_sS --seed S
  R1 solo:      python train_ethics.py --mode solo                 --harm_delivery delayed --out runs/e_r1_solo_sS --seed S
  R1 social:    python train_ethics.py --mode social --virtuous 1  --harm_delivery delayed --out runs/e_r1_virt_sS --seed S

Checkpointing: saves {net, opt, episodes} atomically to <out>/ckpt.pt every batch.
If <out>/ckpt.pt exists at launch, training RESUMES from it automatically (weights,
optimizer state, episode count) and log.csv is appended, so a resubmitted SLURM job
continues to exactly --episodes total. Use --fresh 1 to ignore an existing checkpoint.
--snapshot_every N additionally keeps historical copies ckpt_ep<K>.pt for
harm-over-training curves (~2.7 MB each; default 0 = off).

--wandb mirrors log.csv to Weights & Biases. On a cluster this runs offline
(WANDB_MODE=offline) and is pushed later by slurm/sync_wandb.sh. Chained SLURM
passes reuse the run id kept in <out>/wandb_id.txt and log at
step=<cumulative episodes>, so they append to one run instead of duplicating it.
"""
import argparse, csv, json, os, time
import numpy as np, torch
from ethics import EthicsWorker
from model import SociAPLNet
from ppo import collect, update, HP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["solo", "social", "mixed"], default="social")
    p.add_argument("--aux", choices=["pred", "rec", "none"], default="pred")
    p.add_argument("--virtuous", type=int, default=1)
    p.add_argument("--harm_delivery", choices=["none", "dense", "delayed", "stochastic"], default="none")
    p.add_argument("--harm_lambda", type=float, default=1.0)
    p.add_argument("--n_harm_tiles", type=int, default=6)
    p.add_argument("--episodes", type=int, default=200_000)
    p.add_argument("--batch_episodes", type=int, default=128)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_goals", type=int, default=3)
    p.add_argument("--n_experts", type=int, default=2)
    p.add_argument("--expert_eps", type=float, default=0.0)
    p.add_argument("--p_social", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ethics_debug")
    p.add_argument("--init", default=None, help="explicit checkpoint to start from (e.g. a replication ckpt)")
    p.add_argument("--fresh", type=int, default=0, help="1 = ignore an existing <out>/ckpt.pt and start over")
    p.add_argument("--snapshot_every", type=int, default=0, help="also keep ckpt_ep<K>.pt every N episodes (0 = off)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--wandb", action="store_true", help="mirror log.csv to Weights & Biases")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default=None, help="defaults to the condition, so seeds band together")
    p.add_argument("--wandb_name", default=None, help="defaults to the <out> basename")
    p.add_argument("--wandb_tags", default="", help="comma-separated")
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), **HP}, open(f"{a.out}/config.json", "w"), indent=2)

    kw = dict(harm_delivery=a.harm_delivery, harm_lambda=a.harm_lambda, n_harm_tiles=a.n_harm_tiles)
    workers = [EthicsWorker(a.mode, a.n_experts, a.n_goals, bool(a.virtuous), a.expert_eps,
                            a.p_social, seed=a.seed * 1000 + i, **kw) for i in range(a.n_envs)]
    net = SociAPLNet(aux=a.aux)
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"])
    print(f"params: {sum(x.numel() for x in net.parameters()):,}", flush=True)

    # --- checkpoint restore -------------------------------------------------
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    total = 0
    resume_from = None
    if not a.fresh and os.path.exists(ckpt_path):
        resume_from = ckpt_path
    elif a.init:
        resume_from = a.init
    if resume_from:
        st = torch.load(resume_from, map_location="cpu")
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

    # --- wandb (after the early exit, so over-chained passes log nothing) ----
    run = None
    if a.wandb:
        import wandb_utils
        cond = os.path.basename(a.out.rstrip("/"))            # e.g. e_r0_virt_s0
        run = wandb_utils.init(
            a.out, config={**vars(a), **HP},
            project=a.wandb_project, entity=a.wandb_entity,
            group=a.wandb_group or cond.rsplit("_s", 1)[0],   # e_r0_virt
            name=a.wandb_name or cond,
            job_type="train",
            tags=[a.mode, f"harm_{a.harm_delivery}",
                  "virtuous" if a.virtuous else "shortcut",
                  *a.wandb_tags.split(",")],
        )

    def save_ckpt():
        tmp = ckpt_path + ".tmp"
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "episodes": total}, tmp)
        os.replace(tmp, ckpt_path)  # atomic on POSIX

    # --- logging (append on resume) ----------------------------------------
    log_path = os.path.join(a.out, "log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="", buffering=1)
    log = csv.writer(logf)
    # Named once: wandb mirrors these columns by zipping them against each row,
    # so the dashboard cannot drift from the CSV (see METRICS.md).
    HEADER = ["episodes", "learner_return", "learner_harm", "harm_per_100_moves", "learner_moves",
              "bystander_return", "expert_return", "frac_social",
              "l_pi", "l_v", "l_aux", "ent", "kl", "sec"]
    if new_log:
        log.writerow(HEADER)

    obs = np.stack([w.reset() for w in workers]); h, c = net.init_state(a.n_envs)
    t0 = time.time()
    next_snapshot = ((total // a.snapshot_every) + 1) * a.snapshot_every if a.snapshot_every else None
    while total < a.episodes:
        stats = []
        data, obs, h, c = collect(net, workers, a.batch_episodes, obs, h, c, stats)
        info = update(net, opt, data, a.aux)
        total += len(stats)
        m = lambda k: float(np.mean([s[k] for s in stats]))
        row = [total, m("ep_return"), m("learner_harm"), m("harm_per_100_moves"), m("learner_moves"),
               m("bystander_return"), m("ep_expert_return"), m("social"),
               info.get("pi"), info.get("v"), info.get("aux"), info.get("ent"), info.get("kl"),
               time.time() - t0]
        log.writerow(row); logf.flush()
        if run is not None:
            wandb_utils.log_row(run, HEADER, row)
        print(" ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in row), flush=True)
        save_ckpt()
        if next_snapshot and total >= next_snapshot:
            import shutil
            shutil.copyfile(ckpt_path, os.path.join(a.out, f"ckpt_ep{total}.pt"))
            next_snapshot += a.snapshot_every

    if run is not None:
        wandb_utils.summarize(run, log_path)   # from the whole CSV, so it spans every pass
        wandb_utils.finish(run)


if __name__ == "__main__":
    main()
