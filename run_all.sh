#!/bin/bash
# Submit the whole experiment. Run on a Rorqual LOGIN NODE, from the repo root:
#
#   bash run_all.sh                                        # full grid: 6 conditions x 3 seeds @ 800k
#   GRID=pilot bash run_all.sh                             # 3 conditions x 1 seed @ 200k
#   GRID=pilot PASSES=1 WALLTIME=00:30:00 bash run_all.sh  # 30-min shakedown
#   CHAIN_EVAL=1 bash run_all.sh                           # + evaluation once training finishes
#   bash run_all.sh --eval-only                            # evaluation only, now
#
# This is a SUBMITTER, not a batch script — do not `sbatch` it.
#
# HOW THE CHAINING WORKS. A run needs far more than one walltime to finish, so
# each grid is submitted as PASSES array jobs linked by --dependency=afterany.
# train_ethics.py checkpoints every batch and auto-resumes from ckpt.pt, so a
# pass being killed by the clock is the *expected* way for it to end and the
# next pass picks up exactly where it stopped. Runs that have already reached
# their episode target exit immediately with "nothing to do", so over-chaining
# is harmless — when in doubt, chain more.
#
# afterany, not afterok: the common "failure" here is the walltime running out,
# which must NOT cancel the rest of the chain.
#
# Knobs (all optional):
#   GRID=pilot|full   which grid                            (default full)
#   PASSES=N          chained passes                        (default 1 pilot / 4 full)
#   SEEDS="0 1"       override the seed list
#   FORCE=1           submit finished runs too
#   EXCLUDE=node1,..  pass --exclude straight to sbatch
#   CHAIN_EVAL=1      append the eval array after training
#   plus anything in slurm/env.sh (ACCOUNT, WALLTIME, CPUS, MEM, ...)
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source slurm/env.sh
source slurm/grid.sh

EVAL_ONLY=0
[[ "${1:-}" == "--eval-only" ]] && { EVAL_ONLY=1; shift; }

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run_all.sh is a submitter — run it with bash on a login node, not sbatch." >&2
    exit 1
fi
command -v sbatch >/dev/null || { echo "ERROR: sbatch not found — are you on a login node?" >&2; exit 1; }
[[ -f venv/bin/activate ]] || { echo "ERROR: venv missing. Run: bash slurm/setup_env.sh" >&2; exit 1; }

# SLURM opens the --output file itself, before the job body runs, so logs/ must
# exist at SUBMIT time. A mkdir inside the job script would be too late.
mkdir -p logs "$RUNS_DIR"

# The venv's python is needed below to read episode counts out of ckpt.pt. The
# bare login-node python has no torch, and the skip-check would silently degrade
# to "nothing is finished" and requeue completed runs.
module load $MODULES 2>/dev/null || true
source venv/bin/activate

PASSES="${PASSES:-$([[ "$GRID" == "pilot" ]] && echo 1 || echo 4)}"

SBATCH_OPTS=(
    --account="$ACCOUNT"
    --time="$WALLTIME"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
)
# MAIL_USER is blank unless set in .env; an empty --mail-user makes sbatch
# complain, so only ask for mail when there is somewhere to send it.
[[ -n "$MAIL_USER" ]] && SBATCH_OPTS+=(--mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE")
[[ -n "${EXCLUDE:-}" ]] && { SBATCH_OPTS+=(--exclude="$EXCLUDE"); echo "Excluding nodes: $EXCLUDE"; }

# ── Which indices still need work? ────────────────────────────────────────────
# train_ethics.py would exit immediately on a finished run anyway, but a queued
# no-op still costs a scheduling slot, so skip them here. Reading `episodes` out
# of ckpt.pt is the same check the trainer makes.
episodes_done() {
    local out
    out="$(python - "$1" <<'PY' 2>/dev/null
import sys, os
p = os.path.join(sys.argv[1], "ckpt.pt")
if not os.path.exists(p):
    print(0); raise SystemExit
try:
    import torch
    st = torch.load(p, map_location="cpu", weights_only=True)
    print(int(st.get("episodes", 0)) if isinstance(st, dict) else 0)
except Exception:
    print(0)
PY
    )" || out=""
    # Take only the last line and only if it is a bare integer. Anything else --
    # a warning on stdout, a broken interpreter -- must read as "not finished"
    # rather than crash the submission or wrongly skip a run.
    out="$(tail -n1 <<<"$out")"
    [[ "$out" =~ ^[0-9]+$ ]] && echo "$out" || echo 0
}

echo "Grid '$GRID': $GRID_N runs, $EPISODES episodes each, $PASSES pass(es) of $WALLTIME."
echo ""

TODO=(); SKIPPED=0
for ((i = 0; i < GRID_N; i++)); do
    name="$(grid_run_name "${GRID_CONDS[$i]}" "${GRID_SEEDS[$i]}")"
    if [[ -z "${FORCE:-}" ]]; then
        done_eps="$(episodes_done "$RUNS_DIR/$name")"
        if (( done_eps >= EPISODES )); then
            printf '  [%2d] %-18s %8d/%d episodes -> SKIP (done)\n' "$i" "$name" "$done_eps" "$EPISODES"
            SKIPPED=$((SKIPPED + 1)); continue
        fi
        printf '  [%2d] %-18s %8d/%d episodes\n' "$i" "$name" "$done_eps" "$EPISODES"
    else
        printf '  [%2d] %-18s (FORCE)\n' "$i" "$name"
    fi
    TODO+=("$i")
done

echo ""
if [[ ${#TODO[@]} -eq 0 && "$EVAL_ONLY" -eq 0 ]]; then
    echo "All $GRID_N runs have reached $EPISODES episodes. Nothing to train."
    echo "  FORCE=1 to submit anyway, or CHAIN_EVAL=1 / --eval-only to evaluate."
    if [[ -z "${CHAIN_EVAL:-}" ]]; then
        exit 0
    fi
    echo "  CHAIN_EVAL is set — submitting evaluation only."
    EVAL_ONLY=1
fi

# Comma-separated index list, so a partially-finished grid does not resubmit
# work that is already done.
ARRAY=""
[[ ${#TODO[@]} -gt 0 ]] && ARRAY="$(IFS=,; echo "${TODO[*]}")"
EVAL_ARRAY="$(seq -s, 0 $((GRID_N - 1)))"

# ── Submit ────────────────────────────────────────────────────────────────────
LAST=""
if [[ "$EVAL_ONLY" -eq 0 && ${#TODO[@]} -gt 0 ]]; then
    for ((p = 1; p <= PASSES; p++)); do
        dep=(); [[ -n "$LAST" ]] && dep=(--dependency="afterany:$LAST")
        LAST=$(sbatch --parsable "${SBATCH_OPTS[@]}" "${dep[@]}" \
                      --array="$ARRAY" slurm/train_array.sh)
        printf '  pass %d/%d -> job %s%s\n' "$p" "$PASSES" "$LAST" \
               "$([[ ${#dep[@]} -gt 0 ]] && echo " (after ${dep[0]#--dependency=afterany:})")"
    done
fi

if [[ "$EVAL_ONLY" -eq 1 || -n "${CHAIN_EVAL:-}" ]]; then
    dep=(); [[ -n "$LAST" ]] && dep=(--dependency="afterany:$LAST")
    EVAL_OPTS=(--account="$ACCOUNT" --time="$EVAL_WALLTIME"
               --cpus-per-task="$CPUS" --mem="$MEM")
    [[ -n "$MAIL_USER" ]] && EVAL_OPTS+=(--mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE")
    [[ -n "${EXCLUDE:-}" ]] && EVAL_OPTS+=(--exclude="$EXCLUDE")
    ejid=$(sbatch --parsable "${EVAL_OPTS[@]}" "${dep[@]}" \
                  --array="$EVAL_ARRAY" slurm/eval_array.sh)
    printf '  eval      -> job %s%s\n' "$ejid" "$([[ -n "$LAST" ]] && echo " (after $LAST)")"
fi

echo ""
if [[ "$EVAL_ONLY" -eq 1 ]]; then
    echo "Submitted evaluation for $GRID_N run(s); no training queued."
else
    echo "Submitted ${#TODO[@]} run(s) x $PASSES pass(es); skipped $SKIPPED already done."
fi
echo ""
echo "Monitor:       squeue -u \$USER"
echo "Progress:      tail -f logs/slurm-*.out"
echo "Push to wandb: bash slurm/sync_wandb.sh      # login node; safe to run any time"
echo ""
echo "If a pass runs out of walltime before reaching $EPISODES episodes, just"
echo "re-run this script — it resumes from each run's ckpt.pt and skips the rest."
