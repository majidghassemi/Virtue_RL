#!/bin/bash
# Move code and results between your machine and the cluster. Run LOCALLY.
#
#   bash sync.sh --push                 # code  ->  cluster
#   bash sync.sh --pull                 # results  ->  ./results/
#   bash sync.sh --push -n              # dry run, transfer nothing
#   bash sync.sh --pull --checkpoints   # include the .pt files (GBs)
#
# Set REMOTE_HOST / REMOTE_DIR in .env (REMOTE_DIR must be an ABSOLUTE path).
#
# Authenticates ONCE. DRAC prompts for a password and MFA on EVERY ssh connection,
# so the naive version (ssh mkdir; rsync; ssh chmod) asks three times. Here the
# destination mkdir rides along on the rsync connection via --rsync-path, file
# modes are set locally and carried by rsync -a, and ControlMaster reuses the one
# connection for anything left over.
set -euo pipefail
cd "$(dirname "$0")"
source sociapl/cc_env.sh

MODE=""; DRY=(); CKPT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --push) MODE=push;;
    --pull) MODE=pull;;
    --dry-run|-n) DRY=(--dry-run --itemize-changes);;
    --checkpoints) CKPT=1;;
    *) echo "usage: bash sync.sh --push|--pull [-n] [--checkpoints]" >&2; exit 1;;
  esac
  shift
done
[ -n "$MODE" ] || { echo "usage: bash sync.sh --push|--pull [-n] [--checkpoints]" >&2; exit 1; }
[ -n "$REMOTE_HOST" ] || { echo "ERROR: REMOTE_HOST not set in .env" >&2; exit 1; }
if [ -n "${SLURM_JOB_ID:-}" ]; then echo "ERROR: run this on your local machine." >&2; exit 1; fi

CTL="${TMPDIR:-/tmp}/.ssh-virtue-${USER:-u}"
mkdir -p "$CTL"; chmod 700 "$CTL"
SSH="ssh -o ControlMaster=auto -o ControlPath=$CTL/%r@%h:%p -o ControlPersist=10m"
RSYNC=(rsync -az --info=stats1 -e "$SSH" "${DRY[@]}")

if [ "$MODE" = push ]; then
  echo "push  ->  $REMOTE_HOST:$REMOTE_DIR"
  if [ -f .env ]; then
    chmod 600 .env          # rsync -a carries the mode; no second connection needed
  else
    echo "WARNING: no .env here — the cluster will have no account or wandb key." >&2
  fi
  chmod +x sync.sh sociapl/*.sh 2>/dev/null || true

  # rsync runs --rsync-path on the remote even under --dry-run, so only ask for the
  # mkdir on a real push; a dry run must create nothing.
  MK=()
  [ ${#DRY[@]} -eq 0 ] && MK=(--rsync-path="mkdir -p '$REMOTE_DIR' && rsync")

  # No trailing slashes on directory names: with one, rsync matches real directories
  # only and would happily copy a SYMLINKED venv/ or runs/.
  "${RSYNC[@]}" "${MK[@]}" \
    --exclude='.git' --exclude='venv' --exclude='.venv' --exclude='wheels' \
    --exclude='runs' --exclude='results' --exclude='wandb' --exclude='logs' \
    --exclude='__pycache__' --exclude='*.py[cod]' --exclude='*.egg-info' \
    --exclude='*.pt' --exclude='slurm_*.out' --exclude='*.log' \
    --exclude='.DS_Store' --exclude='.vscode' --exclude='.claude' \
    ./ "$REMOTE_HOST:$REMOTE_DIR/"

  if [ ${#DRY[@]} -eq 0 ]; then
    # .env is a dotfile, so `ls` on the cluster will not show it. Confirm here
    # rather than leave you guessing. Reuses the multiplexed connection: no login.
    echo
    $SSH "$REMOTE_HOST" "cd '$REMOTE_DIR' && ls -la .env 2>/dev/null || echo 'MISSING: .env'"
  fi
  echo
  echo "Done. On the cluster:"
  echo "    cd $REMOTE_DIR/sociapl && bash smoke_test.sh"
else
  EX=(--exclude='__pycache__')
  if [ "$CKPT" -eq 0 ]; then
    EX+=(--exclude='*.pt')
    echo "  (checkpoints excluded — pass --checkpoints to include them)"
  fi
  echo "pull  <-  $REMOTE_HOST:$REMOTE_DIR/sociapl/runs/"
  mkdir -p results
  "${RSYNC[@]}" "${EX[@]}" "$REMOTE_HOST:$REMOTE_DIR/sociapl/runs/" results/
  echo "Done. Results in ./results/"
fi
