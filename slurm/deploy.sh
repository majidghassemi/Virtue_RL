#!/bin/bash
# Ship the code to the cluster with rsync, and pull results back. Run on your
# LOCAL machine, from the repo root:
#
#   bash slurm/deploy.sh                  # push code to the cluster
#   bash slurm/deploy.sh --dry-run        # show what would change, transfer nothing
#   bash slurm/deploy.sh --pull           # bring results back (logs, CSVs, eval JSON)
#   bash slurm/deploy.sh --pull --checkpoints    # ...including the .pt files
#   bash slurm/deploy.sh -y               # skip the confirmation prompt
#
# rsync, not git: no remote, no commit needed to try a change, and it carries
# .env (which is gitignored) so credentials reach the cluster without ever
# entering the repository history.
#
# Configure DRAC_USER / DRAC_CLUSTER / DRAC_REMOTE_DIR in .env (see example.env).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source slurm/env.sh

MODE=push; DRY=0; YES=0; CHECKPOINTS=0; DELETE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pull)        MODE=pull;;
        --push)        MODE=push;;
        --dry-run|-n)  DRY=1;;
        --yes|-y)      YES=1;;
        --checkpoints) CHECKPOINTS=1;;
        --delete)      DELETE=1;;
        -h|--help)     sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0;;
        *) echo "ERROR: unknown option '$1' (see --help)" >&2; exit 1;;
    esac
    shift
done

command -v rsync >/dev/null || { echo "ERROR: rsync not found." >&2; exit 1; }

if [[ -z "$DRAC_USER" ]]; then
    echo "ERROR: DRAC_USER is not set." >&2
    echo "  cp example.env .env  and fill in DRAC_USER / DRAC_CLUSTER / DRAC_REMOTE_DIR." >&2
    exit 1
fi
REMOTE="$DRAC_USER@$DRAC_CLUSTER"

# Never deploy FROM the cluster — that would push the cluster's own copy around.
if [[ -n "${SLURM_JOB_ID:-}" ]] || [[ "$(hostname)" == *alliancecan.ca ]]; then
    echo "ERROR: deploy.sh runs on your local machine, not on the cluster." >&2
    exit 1
fi

# ── What never crosses the wire ───────────────────────────────────────────────
# Outputs, environments and caches. runs/ and logs/ are symlinks into $SCRATCH on
# the cluster; copying over them would replace the symlink with a directory and
# quietly start filling $HOME, which has a file-count quota.
EXCLUDES=(
    --exclude='.git/'
    --exclude='venv/'  --exclude='.venv/'  --exclude='wheels/'
    --exclude='runs/'  --exclude='logs/'   --exclude='wandb/'
    --exclude='__pycache__/' --exclude='*.py[cod]' --exclude='*.egg-info/'
    --exclude='.pytest_cache/' --exclude='.mypy_cache/' --exclude='.ruff_cache/'
    --exclude='*.pt' --exclude='*.ckpt'
    --exclude='.DS_Store' --exclude='.idea/' --exclude='.vscode/' --exclude='.claude/'
    --exclude='results-remote/'
)

RSYNC=(rsync -az --human-readable --info=stats1)
[[ "$DRY" -eq 1 ]] && RSYNC+=(--dry-run --itemize-changes)

echo "── deploy ($MODE) ──────────────────────────────────────"
echo "  local:   $REPO_ROOT"
echo "  remote:  $REMOTE:$DRAC_REMOTE_DIR"
[[ "$DRY" -eq 1 ]] && echo "  DRY RUN — nothing will be transferred"
echo "───────────────────────────────────────────────────────"

confirm() {
    [[ "$YES" -eq 1 || "$DRY" -eq 1 ]] && return 0
    read -r -p "$1 [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
}

if [[ "$MODE" == push ]]; then
    # --delete is opt-in: with output dirs excluded it would still remove any
    # file on the cluster that is absent locally, which is rarely what you want
    # mid-experiment.
    [[ "$DELETE" -eq 1 ]] && { RSYNC+=(--delete); echo "  --delete: remote-only files WILL be removed"; }

    confirm "Push code to $REMOTE:$DRAC_REMOTE_DIR?"

    # Create the destination first; rsync will not make intermediate dirs.
    [[ "$DRY" -eq 0 ]] && ssh "$REMOTE" "mkdir -p $DRAC_REMOTE_DIR"

    "${RSYNC[@]}" "${EXCLUDES[@]}" ./ "$REMOTE:$DRAC_REMOTE_DIR/"

    if [[ "$DRY" -eq 0 ]]; then
        # .env carries the wandb API key, so it must not be world-readable on a
        # shared filesystem. rsync preserves the local mode; enforce 600 anyway.
        if [[ -f .env ]]; then
            ssh "$REMOTE" "chmod 600 $DRAC_REMOTE_DIR/.env" && echo "  .env -> mode 600 on remote"
        else
            echo ""
            echo "  NOTE: no local .env, so none was sent. Credentials and cluster"
            echo "        settings come from it — cp example.env .env and fill it in."
        fi
        chmod_targets="$DRAC_REMOTE_DIR/run_all.sh $DRAC_REMOTE_DIR/slurm/*.sh"
        ssh "$REMOTE" "chmod +x $chmod_targets 2>/dev/null || true"
    fi

    echo ""
    echo "Pushed. Next, on the cluster:"
    echo "    ssh $REMOTE"
    echo "    cd $DRAC_REMOTE_DIR && bash slurm/setup_env.sh    # first time only"
    echo "    GRID=pilot bash run_all.sh"

else
    # Results live on $SCRATCH on the cluster, reached through the runs/ symlink.
    # Follow it with a trailing slash on the source path.
    PULL_EXCLUDES=(--exclude='__pycache__/' --exclude='*.py[cod]')
    if [[ "$CHECKPOINTS" -eq 0 ]]; then
        # Checkpoints are ~2.7 MB each and there are ~20 snapshots per run; the
        # CSVs and eval JSON are what you actually plot.
        PULL_EXCLUDES+=(--exclude='*.pt' --exclude='*.ckpt')
        echo "  (checkpoints excluded — pass --checkpoints to include them)"
    fi

    confirm "Pull results from $REMOTE into ./results-remote/?"
    mkdir -p results-remote
    "${RSYNC[@]}" "${PULL_EXCLUDES[@]}" \
        "$REMOTE:$DRAC_REMOTE_DIR/sociapl/runs/" results-remote/

    echo ""
    echo "Pulled into ./results-remote/"
    echo "  Note: wandb syncing does NOT need this — run slurm/sync_wandb.sh on"
    echo "  the login node instead, which pushes straight to wandb.ai."
fi
