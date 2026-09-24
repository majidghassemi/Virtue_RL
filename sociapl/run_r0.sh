#!/bin/bash
#SBATCH --account=def-YOURPI
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
  echo "python train_ethics.py $ARGS --harm_delivery none --env_config $ENV_CONFIG --episodes 200000 --n_envs 16 --batch_episodes 128 --seed $S --snapshot_every 20000 --out $RUN_ROOT/e_r0_${C}_s${S}"
}

source "$(dirname "$0")/cc_common.sh"
if [ ! -f "$ENV_CONFIG" ]; then
  echo "missing $ENV_CONFIG: tune and freeze the environment first (python summarize_tuning.py --freeze ...)"; exit 1
fi
run_grid
