#!/bin/bash
#SBATCH --account=def-YOURPI   # overridden by cc_sbatch.sh / submit.sh from ../.env
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=0-00:30
#SBATCH --output=slurm_probe_%j.out

# Measure real throughput on a compute node, so walltime and pass count are sized
# from data instead of guesswork. Works either way:
#
#   bash cc_sbatch.sh probe_node.sh          # batch: queue it and read the .out file
#
#   salloc --account=def-mcrowley --gpus-per-node=1 --cpus-per-task=12 \
#          --mem-per-cpu=1500M --time=0:30:00
#   cd $REMOTE_DIR/sociapl && bash probe_node.sh    # interactive
#
# Batch is usually better when GPU nodes are contended: it queues instead of
# holding your terminal, and the resources match the real training jobs exactly.
#
# Add CPU_COMPARE=1 to also time the same work on CPU (doubles the runtime).
set -euo pipefail
# SLURM copies this script into a spool dir before running it, so $0 does NOT
# point into the repo -- resolving paths from it makes the job die instantly.
# SLURM_SUBMIT_DIR is where sbatch was invoked (submit.sh and cc_sbatch.sh always
# invoke from sociapl/). The $0 fallback covers a plain `bash` run.
CC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
[ -f "$CC_DIR/cc_env.sh" ] || CC_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$CC_DIR/cc_env.sh" ] || { echo "ERROR: cannot locate cc_env.sh from $CC_DIR" >&2; exit 1; }
cd "$CC_DIR"
source cc_env.sh
if command -v module >/dev/null 2>&1; then module load StdEnv/2023 python/3.11; fi
VENV=${VENV:-$HOME/venvs/virtue_rl}; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PYTHONWARNINGS=ignore WANDB_MODE=offline

EPISODES=${PROBE_EPISODES:-256}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
PROCS=$(( CPUS - 1 )); [ $PROCS -lt 1 ] && PROCS=1
T=${SLURM_TMPDIR:-$(mktemp -d)}/probe; rm -rf "$T"; mkdir -p "$T"

echo "── node ───────────────────────────────────────────────"
echo "  host            $(hostname)"
echo "  cpus            $CPUS   (env procs: $PROCS)"
echo "  mem/cpu         ${SLURM_MEM_PER_CPU:-?}M"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  | sed 's/^/  gpu             /' || echo "  gpu             none visible"
python - <<'PY'
import torch
from model import get_device
print(f"  torch           {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
print(f"  device auto ->  {get_device('auto')}")
PY

run_probe() {   # $1 = device label
  # Two statements, not `local dev=... out=...$dev`: bash declares every name in a
  # single `local` before assigning any, so the forward reference is unbound under
  # `set -u` and the script dies on the first call.
  local dev="$1"
  local out="$T/$dev"
  echo
  echo "── training $EPISODES episodes on $dev ────────────────"
  # bash's own SECONDS, not /usr/bin/time: that binary is not guaranteed present
  # on a compute node, and the rate below comes from log.csv regardless.
  local t0=$SECONDS
  python train_ethics.py --mode social --virtuous 1 --harm_delivery none \
    --episodes "$EPISODES" --batch_episodes 128 --n_envs 16 \
    --n_procs "$PROCS" --threads 2 --device "$dev" --seed 0 --out "$out" 2>&1 \
    | grep -Ev "Gym|gym|fn\(\)|Users of|migration" | tail -5
  echo "  wallclock       $((SECONDS - t0)) s"
  python - "$out/log.csv" "$dev" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
eps = int(rows[-1]["episodes"]); sec = float(rows[-1]["sec"])
rate = eps / sec
print(f"  -> {rate:.2f} episodes/sec on {sys.argv[2]}")
for target, label in ((800_000, "800k full grid"), (200_000, "200k R0 grid")):
    h = target / rate / 3600
    print(f"     {label:15} {h:6.1f} h/run = {h/24:5.2f} days  "
          f"({-(-h//23.9):.0f} passes of 23:59)")
PY
}

run_probe "$(python -c "from model import get_device; print(get_device('auto'))")"
if [ -n "${CPU_COMPARE:-}" ]; then run_probe cpu; fi

echo
echo "  All 70 runs go in parallel as array tasks, so wall-clock for the grid is"
echo "  ONE run's time (plus queue wait), not the sum."
echo "  Size passes with: bash submit.sh run_all.sh <passes>"
