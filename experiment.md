# Running the experiment on Rorqual

The runbook for the Ethical Goal Cycle grid on the Digital Research Alliance of
Canada. What the numbers mean is in [METRICS.md](METRICS.md); the science is in
[sociapl/README.md](sociapl/README.md).

Everything runs on **CPU**. `sociapl/` contains no CUDA code at all, and the
bottleneck is pure-Python marlgrid env stepping, which a GPU cannot help with.
Never add `--gres` to these jobs.

---

## 0. The shape of it

```
run_all.sh              submitter — the only thing you invoke (login node, bash, NOT sbatch)
slurm/env.sh            account, walltime, paths, wandb settings   ← edit this to retarget
slurm/grid.sh           which conditions x seeds exist, per GRID
slurm/setup_env.sh      one-time venv build (login node)
slurm/train_array.sh    one array task = one (condition, seed)
slurm/eval_array.sh     one array task = every snapshot of one run
slurm/sync_wandb.sh     push offline runs to wandb.ai (login node)
```

Two facts drive the whole design:

- **Compute nodes have no outbound internet.** So `pip install` must happen on a
  login node, and wandb runs offline and is pushed separately.
- **A run needs far more than one walltime.** So training is a *chain* of passes
  that resume from `ckpt.pt`, and being killed by the clock is the normal way for
  a pass to end.

---

## 1. One-time setup

```bash
ssh rorqual.alliancecan.ca
git clone <this-repo> ~/Virtue_RL && cd ~/Virtue_RL
bash slurm/setup_env.sh
```

`setup_env.sh` builds `venv/`, creates the `$SCRATCH/Virtue_RL` output tree,
symlinks `sociapl/runs` and `logs` into it, and smoke-tests the imports. Expect
to see `params 668,555` and `env obs (21, 21, 3)` — if the parameter count is
different, the network does not match the paper and nothing downstream is
comparable.

Then log in to wandb (needed for syncing, not for training):

```bash
source venv/bin/activate && wandb login
```

If your account or email differs from the defaults, edit `slurm/env.sh` — it is
the only file with cluster specifics in it.

### Why output lives on `$SCRATCH`

`$HOME` has a file-**count** quota (~500k inodes) and wandb writes thousands of
small files. `runs/`, `logs/` and `wandb/` therefore live under
`$SCRATCH/Virtue_RL`, reached through symlinks so every script can use plain
relative paths.

**`$SCRATCH` is purged on a ~60-day policy.** Sync to wandb and copy anything you
care about to `~/projects` before then.

---

## 2. Shakedown (30 minutes)

Never submit the full grid first. Prove the plumbing on a short job:

```bash
GRID=pilot PASSES=1 WALLTIME=00:30:00 bash run_all.sh
squeue -u $USER
```

Three tasks should queue, land on CPU nodes, and start writing. Check:

```bash
tail -f logs/slurm-*.out
```

You want to see, in order:

1. the `── Task ──` block naming the run and its condition flags,
2. `params: 668,555`,
3. rows of numbers appearing every ~128 episodes,
4. `[wandb] run <id> (e_r0_virt_s0) mode=offline`.

And on disk: `$SCRATCH/Virtue_RL/runs/e_r0_virt_s0/{log.csv,ckpt.pt,config.json,wandb_id.txt}`.

---

## 3. Pilot (this is the step that sizes everything else)

```bash
GRID=pilot bash run_all.sh
```

3 conditions × 1 seed at 200k episodes: `e_r0_virt`, `e_r0_solo`, `e_r0_short` —
the kill experiment plus its control. [sociapl/README.md](sociapl/README.md) is
explicit that this comes first: *"run a 200k-episode pilot ... and check that
ordering (a)-(d) appears. If it does not, scaling to 1.5M will not rescue it."*

When it finishes, **measure throughput before committing to the full grid**:

```bash
tail -1 $SCRATCH/Virtue_RL/runs/e_r0_virt_s0/log.csv
# columns: episodes,...,sec   ->   episodes/sec = episodes / sec
```

`sec` restarts at each chained pass, so take it from a single pass. Then:

```
PASSES needed  ~=  800000 / (episodes_per_sec * seconds_of_walltime)
```

Set `PASSES` from that number rather than guessing. Over-chaining is harmless
(finished runs exit immediately), under-chaining just means re-running
`run_all.sh` later.

Before scaling, sanity-check the pilot against METRICS.md's *"Reading order when
a run looks wrong"*:

| check | healthy |
|---|---|
| `l_aux` | falls ~0.40 → ~0.10 in the first few hundred episodes |
| `ent` | starts ~1.94 (uniform over 7 actions), eases down |
| `kl` | well under 0.01; repeatedly hitting 0.03 means instability |
| `expert_return` | holds ~20-28 throughout (drift = environment bug) |

If `l_aux` is flat, nothing downstream is interpretable — fix that first.

---

## 4. Full grid

```bash
bash run_all.sh
```

6 conditions × 3 seeds at 800k episodes = 18 array tasks, chained over
`PASSES` (default 4) passes of `WALLTIME` (default 24h).

Knobs:

| variable | effect |
|---|---|
| `GRID=pilot\|full` | which grid (default `full`) |
| `PASSES=N` | chained passes (default 1 pilot / 4 full) |
| `SEEDS="0 1"` | override the seed list |
| `WALLTIME=12:00:00` | per-pass walltime |
| `FORCE=1` | submit finished runs too |
| `EXCLUDE=node1,node2` | passed straight to `sbatch --exclude` |
| `CHAIN_EVAL=1` | append evaluation after training |
| `ACCOUNT`, `CPUS`, `MEM`, ... | anything in `slurm/env.sh` |

**Re-running `run_all.sh` is the normal way to make progress.** It reads each
run's `ckpt.pt`, skips those that have reached their episode target, and builds
the array from only the unfinished indices. So if a pass dies, or the chain ran
out before 800k, just run it again.

To rerun one specific index by hand:

```bash
sbatch --array=7 --account=def-mcrowley --time=24:00:00 \
       --cpus-per-task=8 --mem=8G slurm/train_array.sh
```

`GRID=full bash -c 'source slurm/grid.sh; ...'` will tell you which index is
which; the mapping is printed by `run_all.sh` on every submission.

---

## 5. Push results to wandb

```bash
bash slurm/sync_wandb.sh        # on a LOGIN node
```

Login nodes have outbound internet; compute nodes do not. The script refuses to
run inside a job for that reason.

Safe to run any time, including mid-training — completed passes sync and the
live one goes next time. It is idempotent: wandb marks each offline directory
`.synced` and re-runs are a no-op.

**Chained passes merge into one run.** Each run keeps a stable wandb id in
`runs/<name>/wandb_id.txt`, and metrics are logged at `step=<cumulative
episodes>`, so a later pass appends to the same cloud run instead of creating a
duplicate or overwriting earlier points.

In wandb you get `project=virtue-rl`, one run per `(condition, seed)` named
`e_r0_virt_s0`, grouped by condition so seeds band together. METRICS.md warns:
*"Plot per-seed curves or batch-level histograms, never only the mean"* — the
known failure mode is bimodal seeds, and means hide it.

Each run also carries summary stats: `episodes_to_return_1`,
`final_harm_per_100_moves`, `total_episodes`, `episodes_per_sec`.

---

## 6. Evaluation

```bash
bash run_all.sh --eval-only          # now, on finished runs
CHAIN_EVAL=1 bash run_all.sh         # or queue it behind training
```

Evaluates every `ckpt_ep<K>.pt` snapshot plus the final `ckpt.pt` on the 2×2
grid {teacher, alone} × {seen, unseen}, writing `eval_<ckpt>.json` next to each
checkpoint and logging to a **separate** wandb run named `<run>_eval`. Separate
because evaluating snapshots walks the episode axis from the start again, which
would collide with the training run's already-logged steps.

That sweep is what produces the harm-over-training-time curves, and the
`virtue_gap` values: `harm_per_100_moves` alone minus with teacher. Per
METRICS.md, ~0 means internalised behaviour and > 0 means performative
compliance — *"A large gap is itself a finding (H-C), not a failure."*

`EVAL_EPISODES=100` per cell by default.

---

## 7. Reading the results

The decision comparison is `e_r0_virt` vs `e_r0_solo` vs `e_r0_short`,
teacher-absent, ≥ 3 seeds, on `harm_per_100_moves`. Lower virt than solo, with
short ≥ solo, supports H-A + H-B.

One gate before any of that is meaningful: `learner_return` must cross 1.0 in the
social runs. Solo agents plateau at ~1 (one goal, then stop); exceeding it means
the learner is actually exploiting teacher cues. **If it never crosses, the harm
comparisons are void** — that is a replication-side failure, not an ethics
result.

Full detail in [METRICS.md](METRICS.md).

---

## Gotchas

**Everything must run from `sociapl/`.** The imports there are flat
(`from ethics import ...`). The job scripts `cd sociapl` for you; remember it if
you run anything by hand.

**`$SCRATCH` is purged after ~60 days.** Sync to wandb and copy checkpoints you
need to `~/projects`.

**`runs/` is a symlink** to `$SCRATCH/Virtue_RL/runs`. `ls -l sociapl/runs` if
you are unsure where something landed.

**`gym==0.26.2` failing to install** during setup: it uses legacy setup.py
metadata that newer pip can reject. `setup_env.sh` retries with `setuptools<67`
and `--no-build-isolation` automatically. If it still fails, install it alone and
watch the error — do *not* substitute `gymnasium`, which marlgrid does not target.

**Resumed runs are not bit-reproducible.** Weights, optimizer state and episode
count are checkpointed; `np.random` / torch RNG state are not. A run chained over
4 passes will not match a single-pass run of the same seed step for step. Worth
stating in the write-up, since seeds are reported individually.

**`sec` restarts every pass.** It is per-process wallclock, so do not compute
throughput as `total_episodes / sec` across a chained run. The
`episodes_per_sec` summary in wandb already accounts for this.

**A job that exits immediately with "target episode count already reached;
nothing to do"** is correct behaviour, not an error — that run is finished.
