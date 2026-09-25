# Cluster configuration. Sourced by cc_common.sh (jobs), submit.sh, status.sh,
# sync_wandb.sh and ../sync.sh. Edit ../.env, not this file.
#
# Precedence: shell environment > ../.env > the defaults here.
#
# NOTE for anyone editing: this file is SOURCED, so its last statement becomes the
# return status of `source`. A trailing `[[ ... ]] && ...` that happens to be false
# will therefore kill a `set -e` caller with no error message. Every conditional
# below is a full if-block, and the file ends with `true`.

_cc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || echo "$PWD")"

# .env is parsed, not sourced: it is data, and sourcing would execute whatever it
# contains and clobber variables already set in the environment (including the
# submitting environment that SLURM propagates into a job).
if [ -f "$_cc_root/.env" ]; then
  while IFS= read -r _l || [ -n "$_l" ]; do
    _l="${_l%$'\r'}"; _l="${_l#"${_l%%[![:space:]]*}"}"
    case "$_l" in ''|'#'*) continue;; esac
    _k="${_l%%=*}"; _v="${_l#*=}"
    case "$_k" in
      [A-Za-z_]*) ;;
      *) continue;;
    esac
    if [ -z "${!_k+x}" ]; then export "$_k=$_v"; fi
  done < "$_cc_root/.env"
fi
unset _l _k _v

# ── SLURM ─────────────────────────────────────────────────────────────────────
export ACCOUNT="${ACCOUNT:-def-mcrowley}"
export MAIL_USER="${MAIL_USER:-}"
export MAIL_TYPE="${MAIL_TYPE:-END,FAIL}"

# ── Sync (../sync.sh, run on your local machine) ──────────────────────────────
export REMOTE_HOST="${REMOTE_HOST:-}"
export REMOTE_DIR="${REMOTE_DIR:-/home/prab/links/scratch/virtue_rl}"

# ── Weights & Biases ──────────────────────────────────────────────────────────
# Compute nodes have no outbound internet, so runs are written offline and pushed
# from a login node by sync_wandb.sh.
export WANDB_PROJECT="${WANDB_PROJECT:-virtue_rl}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$_cc_root}"       # wandb appends /wandb
export WANDB_SILENT="${WANDB_SILENT:-true}"
# Keep wandb's scratch files on node-local disk inside a job.
if [ -n "${SLURM_TMPDIR:-}" ]; then
  export WANDB_CACHE_DIR="$SLURM_TMPDIR/wandb_cache"
fi

# sbatch flags derived from the config; used by submit.sh and eval_all.sh.
cc_sbatch_opts() {
  printf '%s\n' "--account=$ACCOUNT"
  if [ -n "$MAIL_USER" ]; then
    printf '%s\n' "--mail-user=$MAIL_USER" "--mail-type=$MAIL_TYPE"
  fi
}

unset _cc_root
true
