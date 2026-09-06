#!/bin/bash
# One array task = one (condition, seed). Submitted by run_all.sh.
#
#SBATCH --job-name=virtue_train
#SBATCH --output=logs/slurm-%A_%a-%x.out
#SBATCH --error=logs/slurm-%A_%a-%x.err
#
# No --gres: this is CPU-only. Resource flags come from run_all.sh.
set -euo pipefail

# SLURM copies this script out of the project directory before running it, so
# BASH_SOURCE is useless here. SLURM_SUBMIT_DIR is where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm/env.sh
source slurm/grid.sh

module load $MODULES
source venv/bin/activate

IDX="${SLURM_ARRAY_TASK_ID:?must run as an array job}"
NAME="$(grid_run_name "${GRID_CONDS[$IDX]}" "${GRID_SEEDS[$IDX]}")"
ARGS="$(grid_cond_args "${GRID_CONDS[$IDX]}")"
NT="${SLURM_CPUS_PER_TASK:-$CPUS}"

# torch spawns its own pool from these; unpinned it oversubscribes the
# allocation and runs slower than single-threaded.
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT"

echo "node $(hostname) | $NAME | $ARGS | $EPISODES episodes | $NT cpus"

# cd sociapl: the imports there are flat (from ethics import ...).
cd sociapl
exec python train_ethics.py $ARGS \
    --episodes "$EPISODES" --snapshot_every "$SNAPSHOT_EVERY" \
    --n_envs 16 --batch_episodes 128 --threads "$NT" \
    --seed "${GRID_SEEDS[$IDX]}" --out "runs/$NAME" \
    --wandb --wandb_tags "$GRID"
