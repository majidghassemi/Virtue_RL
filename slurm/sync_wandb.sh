#!/bin/bash
# Push offline wandb runs to wandb.ai. Run on a Rorqual LOGIN NODE:
#
#   bash slurm/sync_wandb.sh
#
# DRAC login nodes have outbound internet; compute nodes do not. That asymmetry
# is the whole reason training runs with WANDB_MODE=offline and syncing is a
# separate step — there is no need to rsync anything to your laptop first.
#
# Idempotent: wandb drops a .synced marker in each offline dir, and --sync-all
# skips the ones already pushed. Safe to run as often as you like, including
# while training is still going (finished passes sync; the live one syncs next
# time).
#
# Chained SLURM passes of one run share a wandb id (sociapl/wandb_utils.py keeps
# it in <run>/wandb_id.txt), so their separate offline dirs merge into a single
# cloud run rather than appearing as duplicates.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source slurm/env.sh

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: running inside SLURM job $SLURM_JOB_ID — compute nodes have no internet." >&2
    echo "  Run this on a login node instead." >&2
    exit 1
fi

if [[ -f venv/bin/activate ]]; then
    module load $MODULES 2>/dev/null || true
    source venv/bin/activate
fi

command -v wandb >/dev/null || { echo "ERROR: 'wandb' not on PATH (run slurm/setup_env.sh)." >&2; exit 1; }

SYNC_DIR="$WANDB_DIR/wandb"
[[ -d "$SYNC_DIR" ]] || { echo "No wandb directory at $SYNC_DIR — nothing to sync yet."; exit 0; }

TOTAL=$(find "$SYNC_DIR" -maxdepth 1 -type d -name "offline-run-*" 2>/dev/null | wc -l)
PENDING=0
while IFS= read -r d; do
    [[ -f "$d/.synced" ]] || PENDING=$((PENDING + 1))
done < <(find "$SYNC_DIR" -maxdepth 1 -type d -name "offline-run-*" 2>/dev/null)

if [[ "$TOTAL" -eq 0 ]]; then
    echo "No offline runs in $SYNC_DIR."
    echo "  Training writes them there via WANDB_DIR; check that jobs have started."
    exit 0
fi

echo "Offline runs in $SYNC_DIR: $TOTAL total, $PENDING not yet synced."
if [[ "$PENDING" -eq 0 && -z "${FORCE:-}" ]]; then
    echo "Everything is already synced. Nothing to do."
    exit 0
fi

# Credentials are only needed once there is actually something to push, so this
# check comes after the counts — otherwise "nothing to sync" reads as an auth error.
# WANDB_API_KEY normally arrives from .env, which slurm/env.sh sourced above.
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "ERROR: no wandb credentials." >&2
    echo "  Set WANDB_API_KEY in .env (see example.env), then re-deploy:" >&2
    echo "      bash slurm/deploy.sh          # from your local machine" >&2
    echo "  Or, on the cluster:  wandb login  (writes ~/.netrc)" >&2
    exit 1
fi
[[ -n "${WANDB_API_KEY:-}" ]] && echo "Using WANDB_API_KEY from the environment (.env)."

echo "Syncing to project '$WANDB_PROJECT'${WANDB_ENTITY:+ (entity $WANDB_ENTITY)} ..."
echo ""

# Unset offline mode for the sync itself — otherwise the CLI has nowhere to push.
WANDB_MODE=online wandb sync --sync-all \
    --project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} \
    "$SYNC_DIR"

echo ""
echo "Sync complete. View at https://wandb.ai/${WANDB_ENTITY:-<your-entity>}/$WANDB_PROJECT"
