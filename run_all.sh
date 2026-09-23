#!/bin/bash
# Submit the experiment. Run on a cluster LOGIN NODE, from the project directory:
#
#   bash run_all.sh                 # full grid: 6 conditions x 3 seeds @ 800k
#   GRID=pilot bash run_all.sh      # 3 conditions x 1 seed @ 200k
#   PASSES=6 bash run_all.sh        # more chained passes
#
# This is a submitter — run it with bash, not sbatch.
#
# Submits the training array, then an evaluation array that waits for it.
# Training is chained over PASSES jobs because one walltime is not enough:
# train_ethics.py checkpoints every batch and resumes from ckpt.pt, so a pass
# being killed by the clock is normal and the next one continues. Finished runs
# exit immediately, so re-running this script is always safe.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source slurm/env.sh
source slurm/grid.sh

command -v sbatch >/dev/null || { echo "ERROR: no sbatch — run this on a login node." >&2; exit 1; }
[[ -f venv/bin/activate ]] || { echo "ERROR: no venv — run 'bash slurm/setup.sh' first." >&2; exit 1; }

PASSES="${PASSES:-$([[ "$GRID" == pilot ]] && echo 1 || echo 4)}"
mkdir -p logs sociapl/runs      # SLURM opens the --output file before the job runs

OPTS=(--account="$ACCOUNT" --cpus-per-task="$CPUS" --mem="$MEM" --array="0-$((GRID_N - 1))")
[[ -n "$MAIL_USER" ]] && OPTS+=(--mail-user="$MAIL_USER" --mail-type=END,FAIL)

echo "grid '$GRID': $GRID_N runs x $EPISODES episodes, $PASSES pass(es) of $WALLTIME"

# Two jobs writing the same runs/<name>/ would interleave log.csv and race on
# ckpt.pt, silently corrupting both. A log.csv touched in the last 15 minutes
# means a job is very likely still writing there.
live=0
for ((i = 0; i < GRID_N; i++)); do
    name="$(grid_run_name "${GRID_CONDS[$i]}" "${GRID_SEEDS[$i]}")"
    log="sociapl/runs/$name/log.csv"
    mark=""
    if [[ -f "$log" ]] && (( $(date +%s) - $(stat -c %Y "$log") < 900 )); then
        mark="  <-- ACTIVE (written $(( ($(date +%s) - $(stat -c %Y "$log")) / 60 ))m ago)"
        live=$((live + 1))
    fi
    printf '  [%2d] %s%s\n' "$i" "$name" "$mark"
done

if (( live > 0 )) && [[ -z "${FORCE:-}" ]]; then
    echo ""
    echo "ERROR: $live run(s) above are still being written to by a running job." >&2
    echo "  Submitting again would put two processes in the same directory and" >&2
    echo "  corrupt log.csv and ckpt.pt. Options:" >&2
    echo "    - wait for them to finish (bash slurm/status.sh)" >&2
    echo "    - submit only the new seeds, e.g. SEEDS=\"1 2\" bash run_all.sh" >&2
    echo "    - FORCE=1 to override (only if you are certain nothing is running)" >&2
    exit 1
fi
echo ""

# afterany, not afterok: running out of walltime is the expected way for a pass
# to end and must not cancel the rest of the chain.
dep=()
for ((p = 1; p <= PASSES; p++)); do
    jid=$(sbatch --parsable "${OPTS[@]}" --time="$WALLTIME" "${dep[@]}" slurm/train_array.sh)
    echo "  train pass $p/$PASSES -> $jid"
    dep=(--dependency="afterany:$jid")
done

ejid=$(sbatch --parsable "${OPTS[@]}" --time="$EVAL_WALLTIME" "${dep[@]}" slurm/eval_array.sh)
echo "  eval (after $jid) -> $ejid"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     tail -f logs/slurm-*.out"
echo "To wandb: bash slurm/sync_wandb.sh"
