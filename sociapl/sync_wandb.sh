#!/bin/bash
# Push offline wandb runs to wandb.ai. Run on a LOGIN NODE:
#
#   bash sync_wandb.sh
#
# Login nodes have outbound internet; compute nodes do not. That asymmetry is why
# training logs offline and syncing is a separate step.
#
# Idempotent: wandb marks each directory .synced and skips it next time. Safe to run
# while training is still going. Chained passes of one run share a wandb id (kept in
# <run>/wandb_id.txt), so they merge into ONE cloud run rather than N duplicates.
set -euo pipefail
cd "$(dirname "$0")"
source cc_env.sh

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: inside SLURM job $SLURM_JOB_ID — compute nodes have no internet." >&2
  echo "  Run this on a login node." >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then module load StdEnv/2023 python/3.11; fi
VENV=${VENV:-$HOME/venvs/virtue_rl}
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
command -v wandb >/dev/null || { echo "ERROR: wandb not on PATH (run setup_cc.sh)." >&2; exit 1; }

DIR="$WANDB_DIR/wandb"
if [ ! -d "$DIR" ]; then
  echo "nothing to sync: $DIR does not exist yet"; exit 0
fi

total=$(find "$DIR" -maxdepth 1 -type d -name 'offline-run-*' 2>/dev/null | wc -l)
pending=0
while IFS= read -r d; do
  [ -f "$d/.synced" ] || pending=$((pending + 1))
done < <(find "$DIR" -maxdepth 1 -type d -name 'offline-run-*' 2>/dev/null)

echo "$total offline run(s) in $DIR, $pending not yet synced."
if [ "$pending" -eq 0 ] && [ -z "${FORCE:-}" ]; then
  echo "Already up to date."; exit 0
fi

# Credentials are only needed once there is something to push, so this check comes
# after the counts -- otherwise "nothing to sync" reads as an auth failure.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q api.wandb.ai "$HOME/.netrc" 2>/dev/null; then
  echo "ERROR: no wandb credentials." >&2
  echo "  Set WANDB_API_KEY in .env locally, then: bash ../sync.sh --push" >&2
  echo "  Or here: wandb login" >&2
  exit 1
fi

WANDB_MODE=online wandb sync --sync-all --project "$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} "$DIR"

echo "Done: https://wandb.ai/${WANDB_ENTITY:-<entity>}/$WANDB_PROJECT"
