# Running the experiment

Everything lives in one directory on the cluster: `/home/prab/links/scratch/virtue_rl`
(code, venv, runs, logs, wandb). Set it in `.env`.

CPU only — no GPU, ever.

---

## 1. Configure (local, once)

```bash
cp example.env .env
```

Edit `.env`: `REMOTE_HOST`, `REMOTE_DIR`, `ACCOUNT`, `WANDB_API_KEY`.

## 2. Push the code (local)

```bash
bash sync.sh --push
```

Run this again after any code change. Outputs and the venv are never touched.

## 3. Build the venv (login node, once)

```bash
ssh prab@rorqual.alliancecan.ca
cd /home/prab/links/scratch/virtue_rl
bash slurm/setup.sh
```

Must be a login node — compute nodes have no internet.

## 4. Test the environment (login node)

```bash
bash slurm/test.sh
```

Takes ~2 minutes. Trains 128 episodes, resumes to 192, evaluates, and checks the
wandb offline run. Ends with `PASS`. Do not skip this — it catches a broken
install in two minutes instead of after a day in the queue.

## 5. Run (login node)

```bash
GRID=pilot bash run_all.sh      # 3 conditions x 1 seed @ 200k  — do this first
bash run_all.sh                 # full: 6 conditions x 3 seeds @ 800k
```

Submits the training array, then an evaluation array that starts when training
finishes.

```bash
squeue -u $USER
tail -f logs/slurm-*.out
```

## 6. Push to wandb (login node)

```bash
bash slurm/sync_wandb.sh
```

Safe to run any time, including mid-training. Re-running is a no-op.

## 7. Get results (local, optional)

```bash
bash sync.sh --pull             # log.csv, config.json, eval JSON -> ./results/
bash sync.sh --pull --checkpoints
```

---

## Which script runs where

| | where | when |
|---|---|---|
| `sync.sh --push` / `--pull` | your machine | after every code change |
| `slurm/setup.sh` | login node | once |
| `slurm/test.sh` | login node | after setup, and after any dependency change |
| `run_all.sh` | login node | to submit |
| `slurm/sync_wandb.sh` | login node | any time |

Never `sbatch` `run_all.sh` — it is a submitter, run it with `bash`.

---

## Options

| variable | default | |
|---|---|---|
| `GRID` | `full` | `pilot` = 3 conditions x 1 seed @ 200k |
| `PASSES` | 1 pilot / 4 full | chained training jobs |
| `WALLTIME` | `24:00:00` | per pass |
| `SEEDS` | `0 1 2` | e.g. `SEEDS="0 1"` |
| `ACCOUNT`, `CPUS`, `MEM` | see `slurm/env.sh` | |

One walltime is not enough for a full run, so training is chained: each pass
resumes from `ckpt.pt`, and being killed by the clock is normal. Finished runs
exit immediately, so **re-running `run_all.sh` is always safe** — that is how you
make progress if the chain runs out before 800k episodes.

To size `PASSES`, take `episodes` and `sec` from the last row of a run's
`log.csv` after the pilot: `PASSES ≈ 800000 / (episodes/sec × walltime_seconds)`.

---

## If something looks wrong

Read `log.csv` in this order (full detail in [METRICS.md](METRICS.md)):

1. `l_aux` should fall ~0.40 → ~0.10 early. Flat = world model not learning; nothing else is interpretable.
2. `ent` starts ~1.94 and eases down. Pinned or collapsed = exploration problem.
3. `expert_return` should hold ~20-28. Drift = environment bug.
4. `learner_return` must cross 1.0 in social runs, or the harm comparisons are void.
5. Only then read `harm_per_100_moves`, the decision metric.

Other things worth knowing:

- `$SCRATCH` is purged on a ~60-day policy. Sync to wandb before then.
- Resumed runs are not bit-reproducible: weights and episode count are checkpointed, RNG state is not.
- `sec` restarts each pass; don't compute throughput across a chained run.
- "target episode count already reached; nothing to do" means that run is finished, not broken.
