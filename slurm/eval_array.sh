#!/bin/bash
# One array task = every checkpoint of one run, on the 2x2 ethics grid
# {teacher, alone} x {seen, unseen}. Submitted by run_all.sh after training.
#
#SBATCH --job-name=virtue_eval
#SBATCH --output=logs/slurm-%A_%a-%x.out
#SBATCH --error=logs/slurm-%A_%a-%x.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm/env.sh
source slurm/grid.sh

module load $MODULES
source venv/bin/activate

IDX="${SLURM_ARRAY_TASK_ID:?must run as an array job}"
COND="${GRID_CONDS[$IDX]}"
NAME="$(grid_run_name "$COND" "${GRID_SEEDS[$IDX]}")"
NT="${SLURM_CPUS_PER_TASK:-$CPUS}"
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT"

# cd first, then use paths relative to sociapl/ throughout — the imports there
# are flat, so everything has to run from that directory anyway.
cd sociapl
RUN="runs/$NAME"
[[ -d "$RUN" ]] || { echo "no run at sociapl/$RUN — nothing to evaluate"; exit 0; }

# --virtuous picks which teacher accompanies the learner in the "teacher" cells.
# Match it to the training condition so the comparison is honest.
VIRT=1; [[ "$COND" == *short* ]] && VIRT=0

# Ascending episode order: every checkpoint logs into one wandb run walking the
# episode axis forward, so they must arrive in order.
mapfile -t CKPTS < <(ls "$RUN"/ckpt_ep*.pt 2>/dev/null |
    sed 's/.*ckpt_ep\([0-9]*\)\.pt/\1 &/' | sort -n | cut -d' ' -f2- || true)
[[ -f "$RUN/ckpt.pt" ]] && CKPTS+=("$RUN/ckpt.pt")
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "no checkpoints in $RUN"; exit 0; }

echo "$NAME: ${#CKPTS[@]} checkpoints, virtuous=$VIRT"
for c in "${CKPTS[@]}"; do
    b="$(basename "$c" .pt)"; ep="${b#ckpt_ep}"
    step=(); [[ "$ep" =~ ^[0-9]+$ ]] && step=(--wandb_step "$ep")
    echo "── $b ──"
    python eval_ethics.py --ckpt "$c" --virtuous "$VIRT" \
        --episodes "${EVAL_EPISODES:-100}" --out "$RUN/eval_$b.json" \
        --wandb --wandb_dir "$RUN" "${step[@]}"
done
