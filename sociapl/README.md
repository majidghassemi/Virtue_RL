# SociAPL replication — Ndousse, Eck, Levine, Jaques (ICML 2021)

Reimplementation of the learner from *Emergent Social Learning via Multi-agent RL*.
The authors released only the environment (marlgrid); training code is reconstructed
from Sections 3-4 and Appendix 7.4-7.8. Network is 668,555 parameters, matching the paper.

## Setup
    pip install torch numpy    # Linux wheels from PyPI include CUDA; see pytorch.org for other CUDA versions
    cd ../marlgrid && pip install -e . && pip install pyglet    # patched copy, see PATCHES.md
    # PYGLET_HEADLESS=1 is set automatically by envs.py

## Files
    envs.py      make_env, ScriptedExpert (privileged BFS to next correct tile), Worker (solo/social/mixed)
    model.py     SociAPLNet: conv encoder -> FC(576,192) -> LSTM(192) -> policy / value / aux heads
    ppo.py       recurrent PPO-clip + GAE + auxiliary loss; KL target 0.01, hard revert at 0.03
    train.py     training CLI, logs runs/<name>/log.csv
    evaluate.py  H1 (3-goal solo/social) and H2 (4-goal zero-shot) evaluation
    bc.py        behaviour-cloning baseline on expert POV (privileged)

## Reproducing the paper's conditions (Fig. 4, 5, 6, 7)
    # H1, 3-goal Goal Cycle, seeds S = 0..4
    python train.py --mode social --aux pred  --seed S --out runs/social_pred_sS
    python train.py --mode solo   --aux pred  --seed S --out runs/solo_pred_sS
    python train.py --mode social --aux none  --seed S --out runs/social_vanilla_sS
    python train.py --mode social --aux rec   --seed S --out runs/social_rec_sS
    python train.py --mode social --aux pred  --expert_eps 0.3 --seed S --out runs/social_pred_imperfect_sS
    # Sec 4.3: continue a social checkpoint on a 75/25 solo/social mix
    python train.py --mode mixed --aux pred --init runs/social_pred_sS/ckpt.pt --seed S --out runs/mixed_pred_sS
    # H2 and solo transfer
    python evaluate.py --ckpt runs/<run>/ckpt.pt --aux <pred|rec|none> --episodes 100 --out runs/<run>/eval.json
    # IL baseline
    python bc.py --episodes 5000 --epochs 20 --out runs/bc_sS && python evaluate.py --ckpt runs/bc_sS/ckpt.pt --aux none

Defaults match Appendix 7.8: lr 1e-4, gamma 0.993, GAE lambda 0.97, clip 0.2, 128 episodes/batch,
20 minibatches of 512 length-16 segments, loss = pi + 0.1 v + 3 aux + 1e-5 ent, 1.5M episodes.
Env: 13x13, 3 goals, +1 / -1.5, 250 steps, prestige colouring, 21x21x3 partial view.

## Success criterion (decided before running)
Replicated iff across >= 5 seeds: (a) solo and social-vanilla never exceed return 1;
(b) a majority of social aux-pred seeds exceed the expert's return; (c) aux-pred > aux-rec on mean;
(d) 4-goal transfer with experts beats solo transfer. Same signs and ordering, not same decimals.

## Deviations from the paper (state these in the write-up)
1. Experts are scripted (BFS with privileged goal order), not RL-trained via the penalty curriculum.
   A scripted expert never explores, so the paper's "wait, then follow" strategy may not emerge.
   --expert_eps > 0 gives a noisy expert for the imperfect-expert condition.
2. Stored LSTM hidden states and advantages are not recomputed every 2 minibatches.
3. Four Rooms and Maze transfer envs are not in the released repo and are not implemented here.
4. Gradient-norm clipping at 0.5 was added (not mentioned in the paper).
5. Environment runs on gym 0.26 / numpy 2 with the patches listed in ../marlgrid/PATCHES.md.

## What has been verified so far
- Scripted expert: mean return ~26 (3-goal) and ~24 (4-goal) over 250 steps.
- All five training conditions, evaluate.py and bc.py run end to end.
- 112-episode pilot (8 envs, batch 16): aux loss 0.40 -> 0.18, KL within target, no NaNs.
  Learner return ~0 at this scale, as expected; the paper needs ~5e5 episodes for social learning.

## Compute
All scripts take `--device {auto,cuda,cuda:N,mps,cpu}` (default `auto`: CUDA, then Apple MPS, then CPU).
The network, the rollout buffer, GAE and the PPO/BC updates run on that device; checkpoints load onto
it via `map_location`, so GPU-trained checkpoints evaluate on CPU and vice versa.

Env stepping (marlgrid) is pure Python on the CPU. `train.py` / `train_ethics.py` split the
`--n_envs` envs across `--n_procs` subprocesses (default -1 = one per env, up to CPUs-1; 0 =
in-process). Each process steps a fixed slice of envs, so rollouts are identical to in-process
stepping (checked in smoke_test.sh). Measured on a 4-core box, 16 envs: 361 -> 695 steps/s with
3 processes; more cores scale further but that has not been measured here. One run needs about
1.3 GB for the learner plus ~0.5 GB RSS per env process.

### Compute Canada workflow
    bash setup_cc.sh                      # once, on a login node: shared venv ($VENV)
    sbatch smoke_test.sh                  # ~5-10 min on a GPU node; must end with SMOKE TEST PASSED
    TUNE_EPISODES=<N> bash submit.sh tune_env.sh 2
    python summarize_tuning.py runs/tune --freeze runs/tune/<chosen> --out env_frozen.json
    bash submit.sh run_all.sh 4           # 70 array tasks x 4 chained resume passes
    bash submit.sh eval_all.sh 1          # evaluate every finished run (CPU)
    bash sync_wandb.sh                    # push offline wandb runs (login node)

Operator runbook, start to finish: ../instructions.md

Every grid script (tune_env.sh, run_all.sh, run_r0.sh) is one SLURM array: each task gets
1 GPU + 12 cores and runs one training run with 11 env processes. `submit.sh` sets `--array`
from the grid size and chains passes with `afterany`; each pass is < 24 h (the fastest
scheduling class) and resumes from ckpt.pt, so walltime only costs the last unfinished batch.
Knobs (environment variables): `RUNS_PER_JOB=K` packs K runs onto one GPU (the model is
small, so a GPU is mostly idle with one run; give the task K x 12 cores with
`--cpus-per-task`), `THREADS` (torch threads per run, default 2), `DRY_RUN=1` (print commands),
`VENV`, `RUN_ROOT`, `ENV_CONFIG`. Per-run output goes to <out>/stdout.log. Match
`--cpus-per-task` to your cluster's cores per GPU if you want to stay at one GPU's share of a
node; fewer cores just means fewer env processes per run.

The paper reports ~30 h per 1.5M-episode run on 2x1080Ti. Before that, run a 200k-episode pilot,
1 seed x 4 conditions, and check that ordering (a)-(d) appears. If it does not, scaling to 1.5M
will not rescue it.

## Ethics extension (Ethical Goal Cycle)
    ethics.py        HarmTile (shortcut through clutter, -1 to the bystander per traversal),
                     EthicalGoalCycleEnv (R0/R1/R2 via --harm_delivery; layout pool, goal-order
                     shuffle, detour-cost levels), EthicalExpert (virtuous avoids harm tiles in
                     BFS; control variant takes them), EthicsWorker, shared env-design CLI
    train_ethics.py  the experiment grid (see docstring for the 6 commands)
    eval_ethics.py   {teacher, alone} x {seen, unseen_pos, unseen_struct}; reports virtue gaps
    tune_env.sh      SLURM sweep for environment tuning (R0 solo/virt, harm hidden)
    vecenv.py        parallel env stepping (subprocesses; identical rollouts to serial)
    cc_common.sh, submit.sh, setup_cc.sh, smoke_test.sh   Compute Canada job plumbing (see Compute)
    summarize_tuning.py  task-only summary of the sweep; --freeze writes env_frozen.json
    run_all.sh       full grid (70 runs, 5 seeds); refuses to start without env_frozen.json
    run_r0.sh        R0 kill experiment only (15 jobs)
    rerun_evals.sh   one-off: re-evaluates e_r0_solo_s1, e_r1_solo_s1, e_r2_solo_s0 (old round)
    eval_all.sh      evaluates every run under RUN_ROOT; skips evals that already exist
    cc_env.sh        cluster config from ../.env (account, wandb, remote paths)
    status.sh        per-run progress, rate and ETA, read from config.json + log.csv
    sync_wandb.sh    push offline wandb runs to wandb.ai (login node only)
    probe_node.sh    measure episodes/sec on a compute node; sizes walltime and passes
    wandb_utils.py   offline wandb logging; stable run id so chained passes merge

Verified: virtuous teachers cause 0 harm events/ep, shortcut teachers ~26; virtue costs the
teacher ~2 return (22.5 vs 24.6); dense and delayed learner penalties apply correctly.
Kill experiment = e_r0_virt vs e_r0_solo harm rate at ~200k episodes, 5 seeds.

### Environment-design options (train_ethics.py / eval_ethics.py flags or --env_config JSON)
All default to the original environment; with defaults the env is bit-identical to before.

    --penalty P          wrong-order goal penalty (magnitude used; default -1.5)
    --n_goals, --grid_size, --n_harm_tiles
    --view_size V        learner view in tiles (>= 5, default 7 = paper's 21x21). The network
                         adapts; view 7 keeps exactly the paper's 668,555 parameters.
    --harm_detour L      any | zero | small | large. Rejection-samples layouts by detour cost =
                         mean over goal pairs of (shortest harm-free path - shortest path), in
                         moves; zero = virtue is free, small = (0.1, 1], large = [2, inf).
                         Override the band with --detour_band LO HI.
    --n_layouts K        0 = fresh random layout every episode (default); K > 0 = fixed training
                         pool of K layouts (seeds --layout_seed ..+K-1). Needed for unseen_pos.
    --shuffle_order 1    new goal cycle order every episode (use with a layout pool)

Facts measured before choosing these (13x13, 3 goals, 6 harm tiles, 1000 layouts, current
wall-conversion placement):
- Goal positions are already re-drawn every episode (as are clutter and harm tiles), and all
  goal tiles look identical, so the cycle order is already random relative to position. The
  pool/shuffle options only matter once layouts are restricted to a finite pool.
- 78% of layouts already have zero detour; 95th pct 1.33, 99th pct 2.67 moves per pair;
  avoidance is impossible in ~1.4% (excluded from every explicit band). Excess path length
  is always even, so with 3 goals 'small' means exactly one pair with a 2-move detour.
- Cost of rejection sampling per reset: ~4 ms (zero), ~10 ms (small), ~34 ms (large, 3 goals),
  ~125 ms (large, 5 goals). Layouts failing the band after 1000 tries are kept and flagged
  (detour_ok = 0 in log.csv); this has not happened in testing.

### Environment tuning, then freeze (before any full run)
1. `TUNE_EPISODES=<N> bash submit.sh tune_env.sh 2`: R0 solo and R0 virt only, detour
   zero, over penalty {-1.5,-3,-5} x goals {3,4,5} x view {5,7}; `--hide_harm 1` keeps harm
   columns out of log.csv and stdout. Set N from earlier R0-virt curves (long enough for
   return to cross 1); 100k is only a placeholder.
2. `python summarize_tuning.py runs/tune`: task metrics only. A cell passes if solo final
   return <= --solo_max (1.5) in all seeds and virt final return >= --virt_min (2.0) in a
   majority. Fix the thresholds before reading the table.
3. `python summarize_tuning.py runs/tune --freeze runs/tune/<chosen run> --out env_frozen.json`
   writes the frozen env and prints a dated amendment block to paste into HYPOTHESES.md.
4. `run_all.sh` / `run_r0.sh` require env_frozen.json and write to runs/v2/. A resumed run
   refuses to continue if its environment differs from the one it started with.
