#!/bin/bash
# One array task = one (condition, seed) training run.
#
# Not submitted by hand — run_all.sh sizes the array, injects the SBATCH
# resource flags from slurm/env.sh, and chains the passes. To rerun one dead
# index directly:
#     sbatch --array=7 --account=def-mcrowley --time=24:00:00 \
#            --cpus-per-task=8 --mem=8G slurm/train_array.sh
#
# ── SBATCH directives ─────────────────────────────────────────────────────────
# Only the invariant ones live here; account/time/cpus/mem come from run_all.sh
# on the sbatch command line so slurm/env.sh stays the single source of truth.
#SBATCH --job-name=virtue_train
#SBATCH --output=logs/slurm-%A_%a-%x.out
#SBATCH --error=logs/slurm-%A_%a-%x.err
#SBATCH --mail-type=END,FAIL
#
# No --gres: sociapl/ is CPU-only (no torch.device, no .cuda(), no .to()
# anywhere). The bottleneck is pure-Python marlgrid env stepping, which a GPU
# cannot help with.

set -euo pipefail

# SLURM copies this script into /var/spool/slurmd/... before running it, so
# ${BASH_SOURCE[0]} does NOT point into the repo. SLURM_SUBMIT_DIR is the
# directory `sbatch` was invoked from — run_all.sh always submits from the repo
# root. The BASH_SOURCE fallback only covers a direct `bash` invocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

if [[ ! -f venv/bin/activate ]]; then
    echo "ERROR: venv not found at $REPO_ROOT/venv." >&2
    echo "  Build it on a login node first: bash slurm/setup_env.sh" >&2
    exit 1
fi

source slurm/env.sh
source slurm/grid.sh

module load $MODULES
source venv/bin/activate

mkdir -p logs "$RUNS_DIR"

IDX="${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
if (( IDX >= GRID_N )); then
    echo "ERROR: array index $IDX is outside grid '$GRID' (0..$((GRID_N - 1)))." >&2
    exit 1
fi
COND="${GRID_CONDS[$IDX]}"
SEED="${GRID_SEEDS[$IDX]}"
NAME="$(grid_run_name "$COND" "$SEED")"
ARGS="$(grid_cond_args "$COND")"

CPUS_ACTUAL="${SLURM_CPUS_PER_TASK:-$CPUS}"
# torch spawns its own thread pool from these; without pinning them it
# oversubscribes the allocation and runs SLOWER than single-threaded.
export OMP_NUM_THREADS="$CPUS_ACTUAL"
export MKL_NUM_THREADS="$CPUS_ACTUAL"

echo "── Task ───────────────────────────────────────────────"
echo "  node:      $(hostname)"
echo "  grid:      $GRID  (index $IDX of $GRID_N)"
echo "  run:       $NAME"
echo "  args:      $ARGS"
echo "  episodes:  $EPISODES   snapshot_every: $SNAPSHOT_EVERY"
echo "  cpus:      $CPUS_ACTUAL"
echo "  wandb:     $WANDB_MODE -> $WANDB_DIR/wandb  (project $WANDB_PROJECT)"
echo "───────────────────────────────────────────────────────"

# Must run from sociapl/: the imports there are flat (`from ethics import ...`).
# ckpt.pt in the out dir auto-resumes (weights, optimizer, episode count), so a
# walltime kill is the expected way for a pass to end and resubmits are safe.
cd sociapl
exec python train_ethics.py $ARGS \
    --episodes "$EPISODES" \
    --snapshot_every "$SNAPSHOT_EVERY" \
    --n_envs "${N_ENVS:-16}" \
    --batch_episodes "${BATCH_EPISODES:-128}" \
    --threads "$CPUS_ACTUAL" \
    --seed "$SEED" \
    --out "runs/$NAME" \
    --wandb --wandb_tags "$GRID"
