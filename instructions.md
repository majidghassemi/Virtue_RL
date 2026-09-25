# Running the experiment on DRAC

Step-by-step. Design and science are in [sociapl/README.md](sociapl/README.md);
what the logged columns mean is in [METRICS.md](METRICS.md).

Everything on the cluster lives in one directory: `/home/prab/links/scratch/virtue_rl`
(set as `REMOTE_DIR` in `.env`). All job scripts live in and run from `sociapl/`.

---

## 1. Configure (local, once)

```bash
cp example.env .env 2>/dev/null || true   # if .env does not exist yet
$EDITOR .env
```

Needs `REMOTE_HOST`, `REMOTE_DIR`, `ACCOUNT`, and `WANDB_API_KEY`.
`.env` is gitignored; `sync.sh` copies it to the cluster and chmods it 600 there.

Precedence everywhere: shell environment > `.env` > script defaults. So a one-off
`ACCOUNT=def-other bash submit.sh ...` still wins.

## 2. Push the code (local)

```bash
bash sync.sh --push
```

Run again after every local change. Prompts for password + MFA **once**.
Outputs (`runs/`, `wandb/`, `*.pt`) are never pushed, so this is safe mid-run.

## 3. Build the venv (login node, once)

```bash
ssh prab@rorqual.alliancecan.ca
cd /home/prab/links/scratch/virtue_rl/sociapl
bash setup_cc.sh
```

Must be a login node — compute nodes have no internet, so `pip` only works here.

## 4. Smoke test (GPU node)

```bash
bash cc_sbatch.sh smoke_test.sh     # ~5-10 min
```

Use `cc_sbatch.sh`, not bare `sbatch`: the job scripts carry a placeholder
`--account=def-YOURPI`, and this wrapper replaces it with the account from `.env`.
(Grid scripts in steps 6-9 go through `submit.sh`, which does the same thing.)

Must end with `SMOKE TEST PASSED`. It checks the GPU path, that subprocess
rollouts are bit-identical to serial, resume-from-checkpoint, every entry point,
and that harm columns stay blank under `--hide_harm`.

## 5. Measure throughput (GPU node) — sizes everything downstream

```bash
bash cc_sbatch.sh probe_node.sh      # queue it; read slurm_probe_<jobid>.out
```

Or interactively, if GPU nodes are free:

```bash
salloc --account=def-mcrowley --gpus-per-node=1 --cpus-per-task=12 \
       --mem-per-cpu=1500M --time=0:30:00
cd /home/prab/links/scratch/virtue_rl/sociapl && bash probe_node.sh
```

Prints measured episodes/sec and days-per-run for both grids. Use it to pick the
pass count in steps 6 and 8. At 5 ep/s, 800k episodes is ~1.9 days = 2 passes.

## 6. Tune and freeze the environment (required before the full grid)

`run_all.sh` refuses to start without `env_frozen.json`.

```bash
TUNE_EPISODES=<N> bash submit.sh tune_env.sh 2
bash status.sh                                        # RUN_ROOT=runs/tune to watch it
python summarize_tuning.py runs/tune
python summarize_tuning.py runs/tune --freeze runs/tune/<chosen> --out env_frozen.json
```

Set `N` long enough for R0-virt to cross return 1; the 100k default is a
placeholder. Fix `--solo_max` / `--virt_min` before reading the table.
Details: [sociapl/README.md](sociapl/README.md) "Environment tuning, then freeze".

## 7. Train (login node)

```bash
bash submit.sh run_all.sh 4        # 70 tasks x 4 chained passes
# or the R0 kill experiment only:
bash submit.sh run_r0.sh 2         # 15 tasks
```

Each pass is < 24 h (fastest scheduling class) and resumes from `ckpt.pt`, so
running out of walltime is normal. Finished runs exit immediately, so
**over-chaining is free and re-running `submit.sh` is always safe.**

## 8. Monitor

```bash
bash status.sh          # episodes, target, ep/sec, ETA, per run
squeue -u $USER
tail -f runs/v2/<run>/stdout.log
```

## 9. Evaluate (after training)

```bash
bash submit.sh eval_all.sh 1
# add every snapshot, not just the final checkpoint:
EVAL_ALL_SNAPSHOTS=1 bash submit.sh eval_all.sh 1
```

CPU-only, 12 h per task, one task per run. Already-finished evaluations are
skipped, so re-running only fills gaps.

## 10. Push to wandb (login node)

```bash
bash sync_wandb.sh
```

Safe any time, including mid-training. Re-running is a no-op. Chained passes of
one run share a wandb id, so they merge into a single run rather than duplicates.

## 11. Pull results (local)

```bash
bash sync.sh --pull                 # log.csv, config.json, eval JSON -> ./results/
bash sync.sh --pull --checkpoints   # also the .pt files (GBs)
```

---

## Where each script runs

| script | where | when |
|---|---|---|
| `sync.sh --push` / `--pull` | your machine | after every code change |
| `setup_cc.sh` | login node | once |
| `cc_sbatch.sh <job>` | login node | one-off jobs (`smoke_test.sh`) |
| `smoke_test.sh` | GPU node | after setup, after any dependency change |
| `probe_node.sh` | GPU node (`salloc`) | once, to size passes |
| `submit.sh <grid> <passes>` | login node | to submit |
| `status.sh`, `sync_wandb.sh` | login node | any time |

`submit.sh` and `cc_sbatch.sh` are submitters — run them with `bash`, never
`sbatch`. The grid scripts (`run_all.sh`, `tune_env.sh`, `eval_all.sh`) are the
things they submit. Neither needs you to edit the `def-YOURPI` placeholder; both
inject `--account` from `.env`. Bare `sbatch <job>.sh` will fail on that
placeholder.

## Knobs

| variable | default | |
|---|---|---|
| `RUNS_PER_JOB` | 1 | pack K runs on one GPU (give the task K x 12 cores) |
| `THREADS` | 2 | torch threads per run |
| `RUN_ROOT` | `runs/v2` | output root |
| `ENV_CONFIG` | `env_frozen.json` | frozen environment |
| `DRY_RUN=1` | — | print commands instead of running |
| `EXTRA_ARGS` | — | appended to every train command; later flags win |
| `EVAL_EPISODES` | 100 | episodes per evaluation cell |
| `FORCE=1` | — | re-do evaluations that already exist |
| `ACCOUNT`, `MAIL_USER` | from `.env` | injected by `submit.sh` |

Shakedown without submitting anything:

```bash
DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash run_all.sh
bash run_all.sh --count
```

## If something looks wrong

Read `log.csv` in this order (detail in [METRICS.md](METRICS.md)):

1. `l_aux` should fall ~0.40 -> ~0.10 early. Flat = world model not learning; nothing downstream is interpretable.
2. `ent` starts ~1.94 and eases down. Pinned or collapsed = exploration problem.
3. `expert_return` should hold ~20-28. Drift = environment bug.
4. `learner_return` must cross 1.0 in social runs, or the harm comparisons are void.
5. Only then read `harm_per_100_moves`, the decision metric.

Other things worth knowing:

- `$SCRATCH` is purged on a ~60-day policy. Sync to wandb and pull what you need.
- A resumed run refuses to continue if its environment differs from the one it started with — use a new `--out`, or `--fresh 1`.
- "target episode count already reached; nothing to do" means that run is finished, not broken.
- `sec` in `log.csv` restarts each pass; don't compute throughput across a chained run. `status.sh` accounts for this.
- Resumed runs are not bit-reproducible: weights, optimizer and episode count are checkpointed, RNG state is not.
