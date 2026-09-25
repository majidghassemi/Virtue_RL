# Sourced by the SLURM job scripts (run_all.sh, run_r0.sh, tune_env.sh).
# The caller defines: JOBS (array of grid entries) and job_cmd <entry> (prints OUT on line 1,
# the python command on line 2). Knobs (environment variables, exported by sbatch):
#   RUNS_PER_JOB  runs packed into one array task, sharing its GPU and CPUs (default 1)
#   THREADS       torch threads per run (default 2; the GPU does the heavy lifting)
#   EXTRA_ARGS    appended to every train command; later flags win (smoke tests shrink runs)
#   DRY_RUN=1     print the commands for this task instead of running them
#   VENV          virtualenv built by setup_cc.sh (default $HOME/venvs/virtue_rl)
# Cluster config + WANDB_* exports. Sourced before the --count early exit so that
# submit.sh (which calls `bash <grid>.sh --count`) also gets ACCOUNT etc.
source "$(dirname "${BASH_SOURCE[0]}")/cc_env.sh"

K=${RUNS_PER_JOB:-1}
THREADS=${THREADS:-2}

if [ "$1" == "--count" ]; then          # number of array tasks for the grid
  echo $(( (${#JOBS[@]} + K - 1) / K )); exit 0
fi

if [ -z "$DRY_RUN" ]; then
  if command -v module >/dev/null 2>&1; then module load StdEnv/2023 python/3.11; fi
  VENV=${VENV:-$HOME/venvs/virtue_rl}
  if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

run_grid() {
  local task=${SLURM_ARRAY_TASK_ID:?not an array task}
  local cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
  local per=$(( cpus / K ))
  local procs=$(( per - 1 )); [ $procs -lt 1 ] && procs=1   # one core per run stays with the learner
  local pids=() fail=0
  for k in $(seq 0 $((K - 1))); do
    local idx=$(( task * K + k ))
    [ $idx -ge ${#JOBS[@]} ] && break
    mapfile -t spec < <(job_cmd "${JOBS[$idx]}")
    local out=${spec[0]} cmd="${spec[1]} --n_procs $procs --threads $THREADS"
    [ -n "${EXTRA_ARGS:-}" ] && cmd="$cmd $EXTRA_ARGS"
    if [ -n "$DRY_RUN" ]; then echo "[$idx] $cmd"; continue; fi
    mkdir -p "$out"
    echo "[$idx] $cmd   (stdout: $out/stdout.log)"
    $cmd >> "$out/stdout.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || fail=1; done
  return $fail
}
