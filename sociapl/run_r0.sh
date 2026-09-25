#!/bin/bash
#SBATCH --account=def-YOURPI   # overridden by cc_sbatch.sh / submit.sh from ../.env
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=0-23:59
#SBATCH --output=slurm_%x_%A_%a.out

# R0 kill experiment only: {virt, solo, short} x 5 seeds, frozen environment (incl. its detour level).
#   bash submit.sh run_r0.sh 2
SEEDS=(0 1 2 3 4)
ENV_CONFIG=${ENV_CONFIG:-env_frozen.json}
RUN_ROOT=${RUN_ROOT:-runs/v2}   # post-amendment runs; keeps old runs/e_* untouched
JOBS=()
for c in virt solo short; do for s in "${SEEDS[@]}"; do JOBS+=("$c $s"); done; done

job_cmd() {
  read C S <<< "$1"
  case $C in
    virt)  ARGS="--mode social --virtuous 1";;
    short) ARGS="--mode social --virtuous 0";;
    solo)  ARGS="--mode solo";;
  esac
  echo "$RUN_ROOT/e_r0_${C}_s${S}"
  echo "python train_ethics.py $ARGS --harm_delivery none --env_config $ENV_CONFIG --episodes 200000 --n_envs 16 --batch_episodes 128 --seed $S --snapshot_every 20000 --wandb --out $RUN_ROOT/e_r0_${C}_s${S}"
}

# SLURM copies this script into a spool dir before running it, so $0 does NOT
# point into the repo -- resolving paths from it makes the job die instantly.
# SLURM_SUBMIT_DIR is where sbatch was invoked (submit.sh and cc_sbatch.sh always
# invoke from sociapl/). The $0 fallback covers a plain `bash` run.
CC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
[ -f "$CC_DIR/cc_env.sh" ] || CC_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$CC_DIR/cc_env.sh" ] || { echo "ERROR: cannot locate cc_env.sh from $CC_DIR" >&2; exit 1; }
source "$CC_DIR/cc_common.sh"
if [ ! -f "$ENV_CONFIG" ]; then
  echo "missing $ENV_CONFIG: tune and freeze the environment first (python summarize_tuning.py --freeze ...)"; exit 1
fi
run_grid
