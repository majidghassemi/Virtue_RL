#!/bin/bash
# Move code and results between your machine and the cluster. Run LOCALLY.
#
#   bash sync.sh --push          # code  ->  cluster
#   bash sync.sh --pull          # results  ->  ./results/
#   bash sync.sh --push -n       # dry run, transfer nothing
#   bash sync.sh --pull --checkpoints   # include the .pt files
#
# Set REMOTE_HOST and REMOTE_DIR in .env (see example.env).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source slurm/env.sh

MODE=""; DRY=(); CKPT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) MODE=push;;
        --pull) MODE=pull;;
        --dry-run|-n) DRY=(--dry-run --itemize-changes);;
        --checkpoints) CKPT=1;;
        *) echo "usage: bash sync.sh --push|--pull [-n] [--checkpoints]" >&2; exit 1;;
    esac
    shift
done
[[ -n "$MODE" ]] || { echo "usage: bash sync.sh --push|--pull [-n] [--checkpoints]" >&2; exit 1; }
[[ -n "$REMOTE_HOST" ]] || { echo "ERROR: REMOTE_HOST not set — cp example.env .env and fill it in." >&2; exit 1; }

RSYNC=(rsync -az --info=stats1 "${DRY[@]}")

if [[ "$MODE" == push ]]; then
    # Code only. Outputs and the venv stay on the cluster, so pushing mid-run
    # cannot disturb a job in progress.
    echo "push  ->  $REMOTE_HOST:$REMOTE_DIR"
    [[ ${#DRY[@]} -eq 0 ]] && ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
    # No trailing slashes on the directory names: with one, rsync matches real
    # directories only and would happily copy a symlinked venv/ or runs/.
    "${RSYNC[@]}" \
        --exclude='.git' --exclude='venv' --exclude='.venv' \
        --exclude='runs' --exclude='logs' --exclude='wandb' --exclude='results' \
        --exclude='__pycache__' --exclude='*.py[cod]' --exclude='*.egg-info' \
        --exclude='*.pt' --exclude='.DS_Store' --exclude='.vscode' --exclude='.claude' \
        ./ "$REMOTE_HOST:$REMOTE_DIR/"
    if [[ ${#DRY[@]} -eq 0 ]]; then
        # .env holds the wandb key and $HOME is a shared filesystem.
        ssh "$REMOTE_HOST" "chmod 600 '$REMOTE_DIR/.env' 2>/dev/null; chmod +x '$REMOTE_DIR'/*.sh '$REMOTE_DIR'/slurm/*.sh 2>/dev/null; true"
    fi
    echo "Done. On the cluster: cd $REMOTE_DIR && bash run_all.sh"
else
    # Checkpoints are ~2.7 MB each, ~20 per run; the CSVs are what you plot.
    EX=(--exclude='__pycache__/')
    [[ "$CKPT" -eq 0 ]] && EX+=(--exclude='*.pt')
    echo "pull  <-  $REMOTE_HOST:$REMOTE_DIR/sociapl/runs/"
    mkdir -p results
    "${RSYNC[@]}" "${EX[@]}" "$REMOTE_HOST:$REMOTE_DIR/sociapl/runs/" results/
    echo "Done. Results in ./results/"
fi
