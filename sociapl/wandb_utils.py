"""Weights & Biases logging, kept out of the training loop.

METRICS.md is the contract: log.csv is the source of truth and wandb mirrors its
columns, with `episodes` -- not wallclock -- as the x-axis for every chart.

Two things here are load-bearing on a cluster and worth reading before changing:

1. STABLE RUN ID. DRAC walltime kills a job long before 800k episodes, so a run
   is a *chain* of SLURM passes that each resume from ckpt.pt. wandb offline has
   no notion of reopening an existing offline directory -- every pass writes a
   fresh offline-run-*. We persist one id in <out>/wandb_id.txt and reuse it with
   resume="allow", so `wandb sync` folds every pass into a single cloud run.

2. EPISODES AS THE STEP. Logging at step=<cumulative episodes> makes the step
   axis monotonic ACROSS passes, so a later pass can never overwrite an earlier
   pass's points. Using wandb's implicit auto-increment step would restart at 0
   each pass and silently clobber the run's history.

Nothing here may abort training: wandb is a dashboard, not the experiment. Every
entry point degrades to a no-op (returning None) if wandb is missing or errors.
"""
import csv
import os

# Offline runs land in $WANDB_DIR (slurm/env.sh points this at $SCRATCH). Set a
# default so a bare local `--wandb` still works without exporting anything.
os.environ.setdefault("WANDB_MODE", "offline")


def _load():
    try:
        import wandb
        return wandb
    except Exception as e:  # not installed, or a broken install
        print(f"[wandb] unavailable ({e}); continuing without it", flush=True)
        return None


def _run_id(out, id_file="wandb_id.txt"):
    """Read <out>/<id_file>, creating it on first call.

    This file is what ties chained SLURM passes to one cloud run, so it must
    survive in the run directory alongside ckpt.pt -- not be regenerated.
    Training and evaluation keep separate id files: they are separate wandb runs
    because evaluating snapshots replays the episode axis from the start, which
    would collide with the training run's already-logged steps.
    """
    path = os.path.join(out, id_file)
    if os.path.exists(path):
        with open(path) as f:
            rid = f.read().strip()
        if rid:
            return rid
    import wandb
    rid = wandb.util.generate_id()
    with open(path, "w") as f:
        f.write(rid + "\n")
    return rid


def init(out, config, project=None, entity=None, group=None, name=None,
         job_type="train", tags=(), step_metric="episodes", id_file="wandb_id.txt"):
    """Start (or resume) the offline run for <out>. Returns None if unavailable."""
    wandb = _load()
    if wandb is None:
        return None
    try:
        run = wandb.init(
            id=_run_id(out, id_file),
            resume="allow",              # a chained pass reopens the same run
            project=project or os.environ.get("WANDB_PROJECT", "virtue-rl"),
            entity=entity or os.environ.get("WANDB_ENTITY") or None,
            group=group,                 # seeds of one condition band together
            name=name,
            job_type=job_type,
            tags=[t for t in tags if t],
            config=config,
            dir=os.environ.get("WANDB_DIR") or None,
        )
        # Make `episodes` the default x-axis for every panel, per METRICS.md:
        # "The x-axis for all cross-run comparison is `episodes`, not wallclock,
        # so runs of different speeds align."
        wandb.define_metric(step_metric)
        wandb.define_metric("*", step_metric=step_metric)
        print(f"[wandb] run {run.id} ({run.name}) mode={os.environ.get('WANDB_MODE')}", flush=True)
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}); continuing without it", flush=True)
        return None


def log_row(run, header, row):
    """Log one log.csv row.

    Deliberately derived from the CSV header rather than a hand-written key list,
    so the dashboard cannot drift from the file METRICS.md calls the source of
    truth. row[0] is `episodes` and becomes the wandb step.
    """
    log_dict(run, {k: v for k, v in zip(header, row)})


def log_dict(run, d):
    """Log one point. d must carry `episodes`, which becomes the wandb step."""
    if run is None:
        return
    try:
        d = {k: v for k, v in d.items() if v is not None}
        run.log(d, step=int(d["episodes"]))
    except Exception as e:
        print(f"[wandb] log failed ({e})", flush=True)


def summarize(run, log_path):
    """Write the per-run summary statistics METRICS.md asks for.

    Recomputed from the whole of log.csv rather than accumulated in memory, so
    the numbers cover every chained pass, not just the current one.
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

        eps = col("episodes")
        ret = col("learner_return")
        harm = col("harm_per_100_moves")

        # "episodes_to_return_1: first batch where learner_return >= 1.0 (null if
        # never). Operationalises 'did social learning engage, and when'."
        first = None
        for e, r in zip(eps, ret):
            if r is not None and r >= 1.0:
                first = e
                break
        run.summary["episodes_to_return_1"] = first

        # "Final harm_per_100_moves: mean over the last 50 batches."
        tail = [h for h in harm[-50:] if h is not None]
        if tail:
            run.summary["final_harm_per_100_moves"] = sum(tail) / len(tail)

        total = eps[-1] if eps else None
        run.summary["total_episodes"] = total

        # `sec` is wallclock within one process, so it RESETS at every chained
        # SLURM pass. Dividing total episodes by the last `sec` would therefore
        # overstate throughput badly. Measure over the final pass instead: walk
        # back to the last point where `sec` decreased.
        sec = col("sec")
        if sec and sec[-1] is not None:
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
