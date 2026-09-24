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

Environment design (goal penalty, goals, grid, view, detour level, layout pool) comes from
--env_config <frozen.json> plus explicit flags; see ethics.add_env_args. On resume, the
environment in <out>/config.json must match, otherwise the run refuses to continue.
--hide_harm 1 blanks every harm column in log.csv and stdout (blind environment tuning).
"""
import argparse, csv, json, os, time
import numpy as np, torch
from ethics import EthicsWorker, ENV_KEYS, add_env_args, parse_with_env_config, env_kwargs
from model import SociAPLNet, get_device
from ppo import collect, update, HP
from vecenv import make_vec, auto_procs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["solo", "social", "mixed"], default="social")
    p.add_argument("--aux", choices=["pred", "rec", "none"], default="pred")
    p.add_argument("--virtuous", type=int, default=1)
    p.add_argument("--harm_delivery", choices=["none", "dense", "delayed", "stochastic"], default="none")
    p.add_argument("--harm_lambda", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=200_000)
    p.add_argument("--batch_episodes", type=int, default=128)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_experts", type=int, default=2)
    p.add_argument("--expert_eps", type=float, default=0.0)
    p.add_argument("--p_social", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ethics_debug")
    p.add_argument("--init", default=None, help="explicit checkpoint to start from (e.g. a replication ckpt)")
    p.add_argument("--fresh", type=int, default=0, help="1 = ignore an existing <out>/ckpt.pt and start over")
    p.add_argument("--snapshot_every", type=int, default=0, help="also keep ckpt_ep<K>.pt every N episodes (0 = off)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, cuda:1, mps or cpu")
    p.add_argument("--n_procs", type=int, default=-1,
                   help="env subprocesses: -1 = auto (one per env, up to CPUs-1), 0/1 = in-process")
    p.add_argument("--hide_harm", type=int, default=0, help="1 = do not log harm metrics (blind tuning)")
    add_env_args(p)
    a = parse_with_env_config(p)

    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg_path = os.path.join(a.out, "config.json")
    if not a.fresh and os.path.exists(cfg_path) and os.path.exists(os.path.join(a.out, "ckpt.pt")):
        with open(cfg_path) as f:
            old = json.load(f)
        diff = {k: (old[k], getattr(a, k)) for k in ENV_KEYS if k in old and old[k] != getattr(a, k)}
        if diff:
            raise SystemExit(f"environment differs from the run being resumed (old, new): {diff}. "
                             "Use a new --out, or --fresh 1 to discard the old run.")
    with open(cfg_path, "w") as f:
        json.dump({**vars(a), **HP}, f, indent=2)

    kw = dict(harm_delivery=a.harm_delivery, harm_lambda=a.harm_lambda, **env_kwargs(a))
    specs = [((a.mode, a.n_experts), dict(virtuous=bool(a.virtuous), expert_eps=a.expert_eps,
                                          p_social=a.p_social, seed=a.seed * 1000 + i, **kw))
             for i in range(a.n_envs)]
    # env subprocesses are forked before CUDA is initialised (get_device below)
    n_procs = auto_procs(a.n_envs, a.n_procs)
    envs = make_vec(EthicsWorker, specs, n_procs)
    dev = get_device(a.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device: {dev}  env processes: {n_procs if n_procs > 1 else 'in-process'}", flush=True)
    net = SociAPLNet(aux=a.aux, view_size=a.view_size).to(dev)
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
        st = torch.load(resume_from, map_location=dev)
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

    def save_ckpt():
        tmp = ckpt_path + ".tmp"
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "episodes": total}, tmp)
        os.replace(tmp, ckpt_path)  # atomic on POSIX

    # --- logging (append on resume) ----------------------------------------
    cols = ["episodes", "learner_return", "learner_harm", "harm_per_100_moves", "learner_moves",
            "bystander_return", "expert_return", "frac_social",
            "l_pi", "l_v", "l_aux", "ent", "kl", "sec", "detour_cost", "detour_ok"]
    harm_cols = {"learner_harm", "harm_per_100_moves", "bystander_return"}
    log_path = os.path.join(a.out, "log.csv")
    if os.path.exists(log_path):  # resume: keep the existing header (older logs lack the detour columns)
        with open(log_path, newline="") as f:
            cols = next(csv.reader(f), cols)
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="", buffering=1)
    log = csv.DictWriter(logf, fieldnames=cols, extrasaction="ignore")
    if new_log:
        log.writeheader()

    obs = envs.reset(); h, c = net.init_state(a.n_envs)
    t0 = time.time()
    next_snapshot = ((total // a.snapshot_every) + 1) * a.snapshot_every if a.snapshot_every else None
    while total < a.episodes:
        stats = []
        data, obs, h, c = collect(net, envs, a.batch_episodes, obs, h, c, stats)
        info = update(net, opt, data, a.aux)
        total += len(stats)
        m = lambda k: float(np.mean([s[k] for s in stats]))
        finite_detour = [s["detour_cost"] for s in stats if np.isfinite(s["detour_cost"])]
        row = dict(episodes=total, learner_return=m("ep_return"), learner_harm=m("learner_harm"),
                   harm_per_100_moves=m("harm_per_100_moves"), learner_moves=m("learner_moves"),
                   bystander_return=m("bystander_return"), expert_return=m("ep_expert_return"),
                   frac_social=m("social"), l_pi=info.get("pi"), l_v=info.get("v"), l_aux=info.get("aux"),
                   ent=info.get("ent"), kl=info.get("kl"), sec=time.time() - t0,
                   detour_cost=float(np.mean(finite_detour)) if finite_detour else float("nan"),
                   detour_ok=m("detour_ok"))
        if a.hide_harm:
            row.update({k: "" for k in harm_cols})
        log.writerow(row); logf.flush()
        print(" ".join(f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k]) for k in cols), flush=True)
        save_ckpt()
        if next_snapshot and total >= next_snapshot:
            import shutil
            shutil.copyfile(ckpt_path, os.path.join(a.out, f"ckpt_ep{total}.pt"))
            next_snapshot += a.snapshot_every


if __name__ == "__main__":
    main()