"""Weights & Biases logging, kept out of the training loop.

METRICS.md is the contract: log.csv is the source of truth and wandb mirrors its
columns, with `episodes` -- not wallclock -- as the x-axis.

Two things are load-bearing on a cluster:

1. STABLE RUN ID. Walltime kills a job long before the episode target, so a run is
   a *chain* of passes that each resume from ckpt.pt. wandb offline cannot reopen
   an offline directory -- every pass writes a fresh one. We persist one id in
   <out>/wandb_id.txt and reuse it with resume="allow", so `wandb sync` folds all
   passes into a single cloud run instead of N duplicates.

2. EPISODES AS THE STEP. Logging at step=<cumulative episodes> keeps the step axis
   monotonic ACROSS passes. wandb's implicit auto-increment would restart at 0 each
   pass and overwrite the run's history.

Nothing here may abort training: wandb is a dashboard, not the experiment. Every
entry point degrades to a no-op (returns None) if wandb is missing or errors.
"""
import csv
import os

# Compute nodes have no outbound internet; runs are written offline and pushed
# later by sync_wandb.sh from a login node.
os.environ.setdefault("WANDB_MODE", "offline")


def _load():
    try:
        import wandb
        return wandb
    except Exception as e:
        print(f"[wandb] unavailable ({e}); continuing without it", flush=True)
        return None


def _run_id(out, id_file="wandb_id.txt"):
    """Read <out>/<id_file>, creating it on first call.

    This file is what ties chained passes to one cloud run, so it lives in the run
    directory beside ckpt.pt. Training and evaluation keep SEPARATE id files:
    evaluating snapshots replays the episode axis from the start, which would
    collide with the training run's already-logged steps.
    """
    path = os.path.join(out, id_file)
    if os.path.exists(path):
        with open(path) as f:
            rid = f.read().strip()
        if rid:
            return rid
    import wandb
    rid = wandb.util.generate_id()
    os.makedirs(out, exist_ok=True)
    with open(path, "w") as f:
        f.write(rid + "\n")
    return rid


def group_and_name(out):
    """runs/v2/e_r0_virt_d small_s3 -> group 'e_r0_virt_dsmall', name '..._s3'."""
    name = os.path.basename(os.path.normpath(out))
    return name.rsplit("_s", 1)[0], name


def init(out, config, project=None, entity=None, group=None, name=None,
         job_type="train", tags=(), step_metric="episodes", id_file="wandb_id.txt"):
    """Start (or resume) the offline run for <out>. Returns None if unavailable."""
    wandb = _load()
    if wandb is None:
        return None
    try:
        g, n = group_and_name(out)
        run = wandb.init(
            id=_run_id(out, id_file),
            resume="allow",                      # a chained pass reopens the same run
            project=project or os.environ.get("WANDB_PROJECT", "virtue_rl"),
            entity=entity or os.environ.get("WANDB_ENTITY") or None,
            group=group or g,                    # seeds of one condition band together
            name=name or n,
            job_type=job_type,
            tags=[t for t in tags if t],
            config=config,
            dir=os.environ.get("WANDB_DIR") or None,
        )
        # METRICS.md: "The x-axis for all cross-run comparison is `episodes`, not
        # wallclock, so runs of different speeds align."
        wandb.define_metric(step_metric)
        wandb.define_metric("*", step_metric=step_metric)
        print(f"[wandb] {run.id} ({run.name}) mode={os.environ.get('WANDB_MODE')}", flush=True)
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}); continuing without it", flush=True)
        return None


def log_dict(run, d):
    """Log one point. d must carry `episodes`, which becomes the wandb step.
    Blank strings (hide_harm) and NaNs are dropped rather than logged."""
    if run is None:
        return
    try:
        clean = {}
        for k, v in d.items():
            if v is None or v == "":
                continue
            if isinstance(v, float) and v != v:      # NaN
                continue
            clean[k] = v
        run.log(clean, step=int(clean["episodes"]))
    except Exception as e:
        print(f"[wandb] log failed ({e})", flush=True)


def summarize(run, log_path):
    """Per-run summary statistics from METRICS.md's 'Summary statistics per run'.

    Recomputed from the whole of log.csv so the numbers span every chained pass,
    not just the current one.
    """
    if run is None or not os.path.exists(log_path):
        return
    try:
        with open(log_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        def col(name):
            out = []
            for r in rows:
                try:
                    out.append(float(r[name]))
                except (TypeError, ValueError, KeyError):
                    out.append(None)
            return out

        eps, ret, harm = col("episodes"), col("learner_return"), col("harm_per_100_moves")

        # "first batch where learner_return >= 1.0 (null if never)"
        run.summary["episodes_to_return_1"] = next(
            (e for e, r in zip(eps, ret) if r is not None and r >= 1.0), None)

        tail = [h for h in harm[-50:] if h is not None]
        if tail:
            run.summary["final_harm_per_100_moves"] = sum(tail) / len(tail)

        total = eps[-1] if eps else None
        run.summary["total_episodes"] = total

        # `sec` is per-process wallclock and RESETS at every chained pass, so
        # total/last_sec would overstate throughput. Measure over the final pass:
        # walk back to the last point where `sec` decreased.
        sec = col("sec")
        if sec and sec[-1] is not None and total:
            i = len(sec) - 1
            while i > 0 and sec[i - 1] is not None and sec[i - 1] < sec[i]:
                i -= 1
            d_sec = sec[-1] - (sec[i] if i > 0 else 0.0)
            d_eps = eps[-1] - (eps[i] if i > 0 else 0.0)
            if d_sec > 0 and d_eps > 0:
                run.summary["episodes_per_sec"] = d_eps / d_sec
    except Exception as e:
        print(f"[wandb] summarize failed ({e})", flush=True)


def finish(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
