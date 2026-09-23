#!/bin/bash
# Why is throughput low, and is it the thread count or the node?
#
#   bash slurm/bench.sh                  # thread sweep on the solo condition
#   bash slurm/bench.sh --conds          # compare conditions at fixed threads
#   THREADS="1 2 4 8 16" bash slurm/bench.sh
#
# Runs everything sequentially on ONE machine, so differences are the setting
# being varied and not which node SLURM happened to give you. Run it on a
# compute node for numbers that match your jobs:
#
#   salloc --account=$ACCOUNT --cpus-per-task=8 --mem=8G --time=1:00:00
#   bash slurm/bench.sh
#
# On a login node it still works and the RELATIVE ordering is what matters,
# but absolute rates will be noisy -- login nodes are shared.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh
[[ -f venv/bin/activate ]] || { echo "ERROR: no venv — run slurm/setup.sh first." >&2; exit 1; }
module load $MODULES 2>/dev/null || true
source venv/bin/activate

EPISODES="${BENCH_EPISODES:-32}"    # per measurement
N_ENVS="${BENCH_ENVS:-16}"          # matches the real jobs
THREADS="${THREADS:-1 2 4 8}"
BENCH_DIR="sociapl/runs/_bench"
rm -rf "$BENCH_DIR"; trap 'rm -rf "$BENCH_DIR"' EXIT

# One measurement. Prints episodes/sec. --fresh so each is from a cold start.
measure() {  # $1=label  $2=threads  $3...=train_ethics args
    local label="$1" nt="$2"; shift 2
    local rel="runs/_bench/$label-$nt"          # relative to sociapl/
    export OMP_NUM_THREADS="$nt" MKL_NUM_THREADS="$nt"
    ( cd sociapl && python train_ethics.py "$@" \
        --episodes "$EPISODES" --batch_episodes "$EPISODES" --n_envs "$N_ENVS" \
        --threads "$nt" --seed 0 --fresh 1 --out "$rel" >/dev/null 2>&1 ) || {
        printf '  %-12s %3s   FAILED\n' "$label" "$nt"; return; }
    awk -F, -v l="$label" -v t="$nt" 'NR==2 {
        printf "  %-12s %3s  %8.2f ep/s  %8.1f timesteps/s\n", l, t, $1/$NF, ($1*250/'"$N_ENVS"')/$NF }' \
        "sociapl/$rel/log.csv"
}

VIRT="--mode social --virtuous 1 --harm_delivery none"
SOLO="--mode solo --harm_delivery none"
SHORT="--mode social --virtuous 0 --harm_delivery none"

echo "node $(hostname) | $EPISODES episodes, $N_ENVS envs per measurement"
echo "  nproc=$(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo ""

if [[ "${1:-}" == "--conds" ]]; then
    NT="${SLURM_CPUS_PER_TASK:-$CPUS}"
    echo "  CONDITION    thr        rate"
    measure r0_solo  "$NT" $SOLO
    measure r0_virt  "$NT" $VIRT
    measure r0_short "$NT" $SHORT
    echo ""
    echo "  Solo has 2 agents and no expert BFS; social has 4 agents and two BFS"
    echo "  searches per step. Solo SHOULD be fastest. If it is not, the workload"
    echo "  is not what is limiting you."
else
    echo "  CONDITION    thr        rate"
    for t in $THREADS; do measure r0_solo "$t" $SOLO; done
    echo ""
    echo "  The network is tiny (668k params, batch $N_ENVS). If 1-2 threads beats 8,"
    echo "  torch is losing time to thread synchronisation and the jobs should ask"
    echo "  for fewer CPUs: CPUS=2 bash run_all.sh"
fi
