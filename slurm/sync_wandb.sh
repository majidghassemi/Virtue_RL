#!/bin/bash
# Push offline wandb runs to wandb.ai. Run on a LOGIN NODE:
#
#   bash slurm/sync_wandb.sh
#
# Login nodes have internet; compute nodes do not. That is why training logs
# offline and syncing is a separate step.
#
# Idempotent: wandb marks each directory .synced and skips it next time. Safe to
# run while training is still going. Chained passes of one run share a wandb id,
# so they merge into a single run rather than appearing as duplicates.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh

[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: run on a login node — compute nodes have no internet." >&2; exit 1; }
[[ -f venv/bin/activate ]] && { module load $MODULES 2>/dev/null || true; source venv/bin/activate; }

DIR="$WANDB_DIR/wandb"
[[ -d "$DIR" ]] || { echo "nothing to sync: $DIR does not exist yet"; exit 0; }

PENDING=$(find "$DIR" -maxdepth 1 -type d -name 'offline-run-*' ! -exec test -e '{}/.synced' \; -print 2>/dev/null | wc -l)
TOTAL=$(find "$DIR" -maxdepth 1 -type d -name 'offline-run-*' 2>/dev/null | wc -l)
echo "$TOTAL offline run(s), $PENDING not yet synced."
[[ "$PENDING" -gt 0 || -n "${FORCE:-}" ]] || { echo "Already up to date."; exit 0; }

if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q api.wandb.ai "$HOME/.netrc" 2>/dev/null; then
    echo "ERROR: no wandb credentials." >&2
    echo "  Set WANDB_API_KEY in .env locally, then: bash sync.sh --push" >&2
    echo "  Or here: wandb login" >&2
    exit 1
fi

WANDB_MODE=online wandb sync --sync-all --project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} "$DIR"

echo "Done: https://wandb.ai/${WANDB_ENTITY:-<entity>}/$WANDB_PROJECT"
