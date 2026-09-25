#!/bin/bash
# Measure real throughput on a compute node, so walltime and pass count are sized
# from data instead of guesswork.
#
# Allocate a node matching the training jobs, then run this:
#
#   salloc --account=def-mcrowley --gpus-per-node=1 --cpus-per-task=12 \
#          --mem-per-cpu=1500M --time=0:30:00
#   cd $REMOTE_DIR/sociapl && bash probe_node.sh
#
# Add CPU_COMPARE=1 to also time the same work on CPU (doubles the runtime).
set -euo pipefail
cd "$(dirname "$0")"
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
  local dev="$1" out="$T/$dev"
  echo
  echo "── training $EPISODES episodes on $dev ────────────────"
  /usr/bin/time -f "  wallclock       %e s" \
    python train_ethics.py --mode social --virtuous 1 --harm_delivery none \
      --episodes "$EPISODES" --batch_episodes 128 --n_envs 16 \
      --n_procs "$PROCS" --threads 2 --device "$dev" --seed 0 --out "$out" 2>&1 \
    | grep -Ev "Gym|gym|fn\(\)|Users of|migration" | tail -5
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
