#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=0-23:59
#SBATCH --output=slurm_%x_%A_%a.out

# Full experiment grid, 5 seeds per condition, 800k episodes each (70 runs):
#   R0 {virt, solo, short} x detour {zero, small, large}   45 runs  $RUN_ROOT/e_<cond>_d<detour>_s<seed>
#   R2 solo, dense lambda in LAMBDAS                        15 runs  $RUN_ROOT/e_r2_solo_l<lambda>_s<seed>
#   R1 {solo, virt} (delayed), frozen detour level          10 runs  $RUN_ROOT/e_<cond>_s<seed>
#
# Submit with submit.sh (sets --array from the grid and chains resume passes):
#   bash submit.sh run_all.sh 4
# One GPU + 12 cores per array task by default; each run steps its 16 envs in 11 processes.
# RUNS_PER_JOB=K packs K runs onto one GPU (give the task K x 12 cores: --cpus-per-task).
# Each pass runs until walltime (<24 h schedules fastest); ckpt.pt auto-resumes and refuses
# a changed environment. Shakedown: DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash run_all.sh
# Requires the frozen environment: ENV_CONFIG (default env_frozen.json).

SEEDS=(0 1 2 3 4)
R0_CONDS=(r0_virt r0_solo r0_short)
DETOURS=(zero small large)
LAMBDAS=(0.1 0.25 0.5)     # R2 dense penalty, below the per-goal reward of 1
R1_CONDS=(r1_solo r1_virt)
ENV_CONFIG=${ENV_CONFIG:-env_frozen.json}
RUN_ROOT=${RUN_ROOT:-runs/v2}   # post-amendment runs; keeps old runs/e_* untouched

JOBS=()
for c in "${R0_CONDS[@]}"; do for d in "${DETOURS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$c $d - $s"); done; done; done
for l in "${LAMBDAS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("r2_solo - $l $s"); done; done
for c in "${R1_CONDS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$c - - $s"); done; done

job_cmd() {
  read C D L S <<< "$1"
  case $C in
    r0_virt)  ARGS="--mode social --virtuous 1 --harm_delivery none";;
    r0_solo)  ARGS="--mode solo --harm_delivery none";;
    r0_short) ARGS="--mode social --virtuous 0 --harm_delivery none";;
    r2_solo)  ARGS="--mode solo --harm_delivery dense --harm_lambda $L";;
    r1_solo)  ARGS="--mode solo --harm_delivery delayed";;
    r1_virt)  ARGS="--mode social --virtuous 1 --harm_delivery delayed";;
  esac
  OUT=$RUN_ROOT/e_${C}_s${S}
  if [ "$D" != "-" ]; then ARGS="$ARGS --harm_detour $D"; OUT=$RUN_ROOT/e_${C}_d${D}_s${S}; fi
  if [ "$L" != "-" ]; then OUT=$RUN_ROOT/e_${C}_l${L}_s${S}; fi
  echo "$OUT"
  echo "python train_ethics.py $ARGS --env_config $ENV_CONFIG --episodes 800000 --n_envs 16 --batch_episodes 128 --seed $S --snapshot_every 40000 --wandb --out $OUT"
}

source "$(dirname "$0")/cc_common.sh"
if [ ! -f "$ENV_CONFIG" ]; then
  echo "missing $ENV_CONFIG: tune and freeze the environment first (python summarize_tuning.py --freeze ...)"; exit 1
fi
run_grid
