#!/bin/bash
#SBATCH --account=def-YOURPI   # overridden by cc_sbatch.sh / submit.sh from ../.env
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=0-23:59
#SBATCH --output=slurm_%x_%A_%a.out

# Environment tuning sweep (run BEFORE any full run; see README "Environment tuning").
#   - R0 only, solo and virtuous-teacher conditions, harm-free route never longer (--harm_detour zero).
#   - --hide_harm 1: harm columns are blanked in log.csv/stdout; tune on task metrics only.
#   - grid: wrong-goal penalty x number of goals x view size = 3 x 3 x 2 cells, x 2 conditions, 1 seed.
# TUNE_EPISODES must be long enough for R0-virt to cross return 1 under the current env;
# 100k is a placeholder -- set it from your own earlier R0-virt curves:
#   TUNE_EPISODES=150000 bash submit.sh tune_env.sh 2
# Then: python summarize_tuning.py runs/tune

PENALTIES=(-1.5 -3 -5)
GOALS=(3 4 5)
VIEWS=(5 7)
CONDS=(r0_solo r0_virt)
SEED=0
TUNE_EPISODES=${TUNE_EPISODES:-100000}

JOBS=()
for p in "${PENALTIES[@]}"; do for g in "${GOALS[@]}"; do for v in "${VIEWS[@]}"; do for c in "${CONDS[@]}"; do
  JOBS+=("$p $g $v $c"); done; done; done; done

job_cmd() {
  read P G V C <<< "$1"
  case $C in
    r0_solo) ARGS="--mode solo";;
    r0_virt) ARGS="--mode social --virtuous 1";;
  esac
  OUT=${TUNE_ROOT:-runs/tune}/p${P}_g${G}_v${V}_${C}_s${SEED}
  echo "$OUT"
  echo "python train_ethics.py $ARGS --harm_delivery none --harm_detour zero --hide_harm 1 --penalty $P --n_goals $G --view_size $V --episodes $TUNE_EPISODES --n_envs 16 --batch_episodes 128 --seed $SEED --out $OUT"
}

# SLURM copies this script into a spool dir before running it, so $0 does NOT
# point into the repo -- resolving paths from it makes the job die instantly.
# SLURM_SUBMIT_DIR is where sbatch was invoked (submit.sh and cc_sbatch.sh always
# invoke from sociapl/). The $0 fallback covers a plain `bash` run.
CC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
[ -f "$CC_DIR/cc_env.sh" ] || CC_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$CC_DIR/cc_env.sh" ] || { echo "ERROR: cannot locate cc_env.sh from $CC_DIR" >&2; exit 1; }
source "$CC_DIR/cc_common.sh"
run_grid
