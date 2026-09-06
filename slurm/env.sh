# Cluster configuration — the ONE file to edit when retargeting.
#
# Sourced (never executed) by run_all.sh, setup_env.sh, train_array.sh,
# eval_array.sh and sync_wandb.sh. Everything cluster-specific lives here so no
# account name or path is copy-pasted into a job script.
#
# Precedence, highest first:
#   1. the environment      ACCOUNT=def-other bash run_all.sh
#   2. .env                 credentials and per-user settings (gitignored)
#   3. the defaults below
#
# ── .env ──────────────────────────────────────────────────────────────────────
# Credentials and per-user settings live in .env, never in git. Copy example.env
# to .env and fill it in.
#
# Loaded FIRST so its values are in place before the ${VAR:-default} lines below,
# which is what makes .env beat the defaults. But a variable ALREADY set in the
# environment is left alone, so a one-off `ACCOUNT=def-other bash run_all.sh`
# still wins -- and so does the submitting environment that SLURM propagates into
# a job. Parsed by hand rather than `source`d: .env is data, and sourcing it would
# execute whatever it contains and clobber the shell env unconditionally.
_ENV_SELF="${BASH_SOURCE[0]}"
_ENV_ROOT="$(cd "$(dirname "$_ENV_SELF")/.." 2>/dev/null && pwd || echo "$PWD")"
ENV_FILE="${ENV_FILE:-$_ENV_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r _line || [[ -n "$_line" ]]; do
        _line="${_line%$'\r'}"                       # tolerate CRLF
        [[ "$_line" =~ ^[[:space:]]*(#|$) ]] && continue
        _line="${_line#"${_line%%[![:space:]]*}"}"   # ltrim
        _line="${_line#export }"
        _key="${_line%%=*}"; _key="${_key%"${_key##*[![:space:]]}"}"
        [[ "$_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -n "${!_key+x}" ]] && continue            # already set: leave it
        _val="${_line#*=}"
        _val="${_val#"${_val%%[![:space:]]*}"}"      # trim
        _val="${_val%"${_val##*[![:space:]]}"}"
        [[ "$_val" == \"*\" || "$_val" == \'*\' ]] && _val="${_val:1:${#_val}-2}"
        export "$_key=$_val"
    done < "$ENV_FILE"
fi
unset _ENV_SELF _ENV_ROOT _line _key _val

# ── SLURM ─────────────────────────────────────────────────────────────────────
ACCOUNT="${ACCOUNT:-def-mcrowley}"
MAIL_USER="${MAIL_USER:-}"
MAIL_TYPE="${MAIL_TYPE:-END,FAIL}"

# CPU-only: sociapl/ has no CUDA code at all (no torch.device, no .cuda(), no
# .to()). The bottleneck is pure-Python marlgrid env stepping, which a GPU would
# not help with. Never add --gres here.
CPUS="${CPUS:-8}"          # matches train_ethics.py --threads
MEM="${MEM:-8G}"           # the 668k-param net is tiny; this is env + rollout buffers

# 24h rather than the 7-day maximum: short asks backfill into maintenance gaps
# and start far sooner on a busy cluster. Total episodes are reached by chaining
# PASSES of them (ckpt.pt auto-resumes), not by one long job.
WALLTIME="${WALLTIME:-24:00:00}"
EVAL_WALLTIME="${EVAL_WALLTIME:-03:00:00}"

# ── Modules ───────────────────────────────────────────────────────────────────
# python/3.11, not 3.12: both requirements files pin against 3.11 and that is
# where gym==0.26.2 / gym-minigrid==1.2.2 / numba are known-good together.
# No cuda/cudnn — see above.
MODULES="${MODULES:-StdEnv/2023 python/3.11}"

# ── Paths ─────────────────────────────────────────────────────────────────────
# Code in $HOME (backed up, small); all output on $SCRATCH. $HOME has a
# file-COUNT quota (~500k inodes) and wandb writes thousands of small files, so
# runs/ logs/ and wandb/ must not live there.
#
# $SCRATCH is not set outside DRAC; fall back to a local dir so the scripts are
# testable off-cluster.
SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH:-$HOME/scratch}/Virtue_RL}"
RUNS_DIR="${RUNS_DIR:-$SCRATCH_ROOT/runs}"
LOGS_DIR="${LOGS_DIR:-$SCRATCH_ROOT/logs}"

# ── Cluster (used by slurm/deploy.sh on your LOCAL machine) ──────────────────
DRAC_USER="${DRAC_USER:-}"
DRAC_CLUSTER="${DRAC_CLUSTER:-rorqual.alliancecan.ca}"
DRAC_REMOTE_DIR="${DRAC_REMOTE_DIR:-~/Virtue_RL}"

# ── Weights & Biases ──────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-virtue_rl}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"          # blank = your default entity

# Compute nodes have NO outbound internet on DRAC. Offline runs are written to
# WANDB_DIR and pushed later by slurm/sync_wandb.sh from a login node.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT}"   # wandb appends /wandb itself
export WANDB_SILENT="${WANDB_SILENT:-true}"

# Keep wandb's scratch files on node-local disk, off the shared filesystem.
# $SLURM_TMPDIR only exists inside a job; outside one, leave the defaults alone.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export WANDB_CACHE_DIR="$SLURM_TMPDIR/wandb_cache"
    export WANDB_CONFIG_DIR="$SLURM_TMPDIR/wandb_config"
fi
