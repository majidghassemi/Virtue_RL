#!/bin/bash
# One array task = the full snapshot sweep for one training run.
#
# Evaluates every ckpt_ep<K>.pt in ascending episode order, plus the final
# ckpt.pt, on the 2x2 ethics grid {teacher, alone} x {seen, unseen}. Logging
# each at its own episode count is what produces the harm-over-training-time
# curves METRICS.md asks for.
#
# Submitted by run_all.sh with CHAIN_EVAL=1, or by hand once training is done:
#     bash run_all.sh --eval-only
#
# ── SBATCH directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=virtue_eval
#SBATCH --output=logs/slurm-%A_%a-%x.out
#SBATCH --error=logs/slurm-%A_%a-%x.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

[[ -f venv/bin/activate ]] || { echo "ERROR: venv not found at $REPO_ROOT/venv (run slurm/setup_env.sh)." >&2; exit 1; }

source slurm/env.sh
source slurm/grid.sh

module load $MODULES
source venv/bin/activate
mkdir -p logs

IDX="${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
(( IDX < GRID_N )) || { echo "ERROR: index $IDX outside grid '$GRID'." >&2; exit 1; }
COND="${GRID_CONDS[$IDX]}"
SEED="${GRID_SEEDS[$IDX]}"
NAME="$(grid_run_name "$COND" "$SEED")"
RUN_DIR="$RUNS_DIR/$NAME"

[[ -d "$RUN_DIR" ]] || { echo "Nothing to evaluate: $RUN_DIR does not exist."; exit 0; }

CPUS_ACTUAL="${SLURM_CPUS_PER_TASK:-$CPUS}"
export OMP_NUM_THREADS="$CPUS_ACTUAL" MKL_NUM_THREADS="$CPUS_ACTUAL"

# The learner's own virtue flag is a training-time property of its TEACHER; at
# eval time --virtuous only picks which teacher accompanies it in the "teacher"
# cells. Keep it matched to the training condition so the comparison is honest.
VIRTUOUS=1
[[ "$COND" == *short* ]] && VIRTUOUS=0

# Ascending episode order: eval logs into one wandb run walking the episode
# axis forward, so the points must arrive in order.
mapfile -t SNAPS < <(ls "$RUN_DIR"/ckpt_ep*.pt 2>/dev/null \
    | sed 's/.*ckpt_ep\([0-9]*\)\.pt/\1 &/' | sort -n | cut -d' ' -f2- || true)
[[ -f "$RUN_DIR/ckpt.pt" ]] && SNAPS+=("$RUN_DIR/ckpt.pt")

if [[ ${#SNAPS[@]} -eq 0 ]]; then
    echo "Nothing to evaluate: no checkpoints in $RUN_DIR."; exit 0
fi

echo "── Eval $NAME (${#SNAPS[@]} checkpoints, virtuous=$VIRTUOUS) ──"
cd sociapl
for ckpt in "${SNAPS[@]}"; do
    ep="$(basename "$ckpt" .pt)"; ep="${ep#ckpt_ep}"
    step_arg=()
    # The final ckpt.pt carries its own episode count; snapshots are named by it.
    [[ "$ep" =~ ^[0-9]+$ ]] && step_arg=(--wandb_step "$ep")
    echo ""
    echo "── $(basename "$ckpt") ──"
    python eval_ethics.py \
        --ckpt "$ckpt" \
        --virtuous "$VIRTUOUS" \
        --episodes "${EVAL_EPISODES:-100}" \
        --out "$RUN_DIR/eval_$(basename "$ckpt" .pt).json" \
        --wandb --wandb_dir "$RUN_DIR" "${step_arg[@]}"
done

echo ""
echo "Done: ${#SNAPS[@]} checkpoints evaluated for $NAME."
