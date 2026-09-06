# Shared config. Sourced by every other script. Edit .env, not this file.
#
# Precedence: shell environment > .env > the defaults here.

# ── .env ──────────────────────────────────────────────────────────────────────
# Parsed rather than sourced: it is data, and sourcing would execute it and
# overwrite variables already set in the environment.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || echo "$PWD")"
if [[ -f "$_root/.env" ]]; then
    while IFS= read -r _l || [[ -n "$_l" ]]; do
        _l="${_l%$'\r'}"; _l="${_l#"${_l%%[![:space:]]*}"}"
        [[ "$_l" =~ ^(#|$) ]] && continue
        _k="${_l%%=*}"; _v="${_l#*=}"
        [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -n "${!_k+x}" ]] && continue          # already set: leave it
        export "$_k=$_v"
    done < "$_root/.env"
fi
unset _l _k _v

# ── Where ─────────────────────────────────────────────────────────────────────
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-/home/prab/links/scratch/virtue_rl}"
REPO_ROOT="$_root"
unset _root

# ── SLURM ─────────────────────────────────────────────────────────────────────
# CPU only. sociapl/ has no CUDA code; the bottleneck is pure-Python env
# stepping, which a GPU cannot help. Never add --gres.
ACCOUNT="${ACCOUNT:-def-mcrowley}"
MAIL_USER="${MAIL_USER:-}"
CPUS="${CPUS:-8}"
MEM="${MEM:-8G}"
WALLTIME="${WALLTIME:-24:00:00}"
EVAL_WALLTIME="${EVAL_WALLTIME:-03:00:00}"
MODULES="${MODULES:-StdEnv/2023 python/3.11}"

# ── Weights & Biases ──────────────────────────────────────────────────────────
# Compute nodes have no internet, so runs are written offline and pushed later
# by slurm/sync_wandb.sh from a login node.
export WANDB_PROJECT="${WANDB_PROJECT:-virtue_rl}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT}"      # wandb appends /wandb
export WANDB_SILENT="${WANDB_SILENT:-true}"

# Keep wandb's scratch files on node-local disk inside a job. Written as an if,
# not `[[ ]] && ...`: this file is SOURCED, so a false test on the last line
# would become the return status of `source` and kill a `set -e` caller.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export WANDB_CACHE_DIR="$SLURM_TMPDIR/wandb_cache"
fi
