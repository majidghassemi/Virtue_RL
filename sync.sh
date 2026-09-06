#!/bin/bash
# Move code and results between your machine and the cluster. Run LOCALLY.
#
#   bash sync.sh --push          # code  ->  cluster
#   bash sync.sh --pull          # results  ->  ./results/
#   bash sync.sh --push -n       # dry run, transfer nothing
#   bash sync.sh --pull --checkpoints   # include the .pt files
#
# Set REMOTE_HOST and REMOTE_DIR in .env (see example.env). REMOTE_DIR must be
# an absolute path, not ~/... — it is quoted when sent to the remote shell.
#
# Authenticates ONCE. DRAC prompts for a password and MFA on every SSH
# connection, so this opens a single multiplexed connection and reuses it for
# everything; ControlPersist keeps it briefly so a follow-up push or your own
# `ssh` is free too. Anything that would need a second connection (mkdir, chmod)
# is folded into the rsync call or done locally instead.
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

CTL_DIR="${TMPDIR:-/tmp}/.ssh-virtue-${USER:-u}"
mkdir -p "$CTL_DIR"; chmod 700 "$CTL_DIR"
SSH_CMD="ssh -o ControlMaster=auto -o ControlPath=$CTL_DIR/%r@%h:%p -o ControlPersist=10m"

RSYNC=(rsync -az --info=stats1 -e "$SSH_CMD" "${DRY[@]}")

if [[ "$MODE" == push ]]; then
    # Code only. Outputs and the venv stay on the cluster, so pushing mid-run
    # cannot disturb a job in progress.
    echo "push  ->  $REMOTE_HOST:$REMOTE_DIR"

    if [[ -f .env ]]; then
        # Set locally; rsync -a carries the mode across, so no second connection
        # is needed to secure it on the far side.
        chmod 600 .env
    else
        echo "WARNING: no .env in $(pwd) — nothing to carry your cluster settings" >&2
        echo "         or wandb key. cp example.env .env, fill it in, push again." >&2
    fi
    chmod +x sync.sh run_all.sh slurm/*.sh 2>/dev/null || true

    # --rsync-path creates the destination as part of THIS connection instead of
    # a separate `ssh mkdir`. Skipped under --dry-run: rsync runs rsync-path on
    # the remote regardless of --dry-run, and a dry run must not create anything.
    MKDIR=()
    [[ ${#DRY[@]} -eq 0 ]] && MKDIR=(--rsync-path="mkdir -p '$REMOTE_DIR' && rsync")

    # No trailing slashes on the directory names: with one, rsync matches real
    # directories only and would happily copy a symlinked venv/ or runs/.
    "${RSYNC[@]}" "${MKDIR[@]}" \
        --exclude='.git' --exclude='venv' --exclude='.venv' \
        --exclude='runs' --exclude='logs' --exclude='wandb' --exclude='results' \
        --exclude='__pycache__' --exclude='*.py[cod]' --exclude='*.egg-info' \
        --exclude='*.pt' --exclude='.DS_Store' --exclude='.vscode' --exclude='.claude' \
        ./ "$REMOTE_HOST:$REMOTE_DIR/"

    if [[ ${#DRY[@]} -eq 0 ]]; then
        # Confirm what actually landed. .env is a dotfile, so a plain `ls` on the
        # cluster will not show it -- check here instead of wondering. Reuses the
        # multiplexed connection, so it costs no extra login.
        echo ""
        $SSH_CMD "$REMOTE_HOST" "cd '$REMOTE_DIR' && ls -la .env 2>/dev/null || echo 'MISSING: .env'"
    fi
    echo ""
    echo "Done. On the cluster:"
    echo "    cd $REMOTE_DIR && bash slurm/test.sh"
else
    # Checkpoints are ~2.7 MB each, ~20 per run; the CSVs are what you plot.
    EX=(--exclude='__pycache__')
    [[ "$CKPT" -eq 0 ]] && EX+=(--exclude='*.pt')
    echo "pull  <-  $REMOTE_HOST:$REMOTE_DIR/sociapl/runs/"
    mkdir -p results
    "${RSYNC[@]}" "${EX[@]}" "$REMOTE_HOST:$REMOTE_DIR/sociapl/runs/" results/
    echo "Done. Results in ./results/"
fi
