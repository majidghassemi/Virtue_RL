#!/bin/bash
#SBATCH --account=def-YOURPI   # overridden by cc_sbatch.sh / submit.sh from ../.env
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-12:00
#SBATCH --output=slurm_%x_%A_%a.out

# Evaluate every finished run in $RUN_ROOT on {teacher, alone} x {seen, unseen_pos,
# unseen_struct}. One array task per run.
#
#   bash submit.sh eval_all.sh 1              # after training finishes
#   EVAL_ALL_SNAPSHOTS=1 bash submit.sh eval_all.sh 1    # every ckpt_ep*.pt too
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash eval_all.sh
#
# CPU only: evaluation is batch-1 inference and gains nothing from a GPU, so these
# tasks schedule far sooner than the GPU training jobs.
#
# 12 h, not 3 h. The final checkpoint alone is 6 conditions x --eval_episodes; with
# EVAL_ALL_SNAPSHOTS it is that times ~20 snapshots. A 3 h limit silently truncated
# this in an earlier round and left runs with no final eval at all -- if a task hits
# the wall now, re-running is safe because finished evals are skipped.

RUN_ROOT=${RUN_ROOT:-runs/v2}
EVAL_EPISODES=${EVAL_EPISODES:-100}

# The grid is whatever exists on disk, so this stays correct no matter which grid
# script produced the runs.
shopt -s nullglob
JOBS=()
for d in "$RUN_ROOT"/*/; do
  [ -f "${d}ckpt.pt" ] && JOBS+=("${d%/}")
done

# No job_cmd/run_grid here: cc_common.sh is sourced only for --count, the venv and
# the WANDB_* exports. This script drives its own loop because one run maps to a
# variable number of checkpoints.
# SLURM copies this script into a spool dir before running it, so $0 does NOT
# point into the repo -- resolving paths from it makes the job die instantly.
# SLURM_SUBMIT_DIR is where sbatch was invoked (submit.sh and cc_sbatch.sh always
# invoke from sociapl/). The $0 fallback covers a plain `bash` run.
CC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
[ -f "$CC_DIR/cc_env.sh" ] || CC_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$CC_DIR/cc_env.sh" ] || { echo "ERROR: cannot locate cc_env.sh from $CC_DIR" >&2; exit 1; }
source "$CC_DIR/cc_common.sh"

if [ ${#JOBS[@]} -eq 0 ]; then
  echo "no runs with ckpt.pt under $RUN_ROOT"; exit 1
fi

task=${SLURM_ARRAY_TASK_ID:?not an array task}
K=${RUNS_PER_JOB:-1}
fail=0
for k in $(seq 0 $((K - 1))); do
  idx=$(( task * K + k ))
  [ $idx -ge ${#JOBS[@]} ] && break
  run="${JOBS[$idx]}"

  # Final checkpoint always; snapshots only on request, in ascending episode order
  # so the eval wandb run walks the episode axis forward.
  ckpts=()
  if [ -n "${EVAL_ALL_SNAPSHOTS:-}" ]; then
    while IFS= read -r c; do ckpts+=("$c"); done < <(
      ls "$run"/ckpt_ep*.pt 2>/dev/null |
      sed 's/.*ckpt_ep\([0-9]*\)\.pt/\1 &/' | sort -n | cut -d' ' -f2-)
  fi
  ckpts+=("$run/ckpt.pt")

  for c in "${ckpts[@]}"; do
    b=$(basename "$c" .pt)
    outjson="$run/eval_${b}.json"
    if [ -f "$outjson" ] && [ -z "${FORCE:-}" ]; then
      echo "[$idx] skip $(basename "$run")/$b (done)"; continue
    fi
    # An array, not a string: paths keep their spaces and nothing depends on
    # word-splitting behaving.
    cmd=(python eval_ethics.py --ckpt "$c" --episodes "$EVAL_EPISODES" --device cpu
         --out "$outjson" --wandb --wandb_dir "$run")
    ep=${b#ckpt_ep}
    case "$ep" in ''|*[!0-9]*) ;; *) cmd+=(--wandb_step "$ep");; esac
    if [ -n "${DRY_RUN:-}" ]; then echo "[$idx] ${cmd[*]}"; continue; fi
    echo "[$idx] $(basename "$run")/$b -> $outjson"
    "${cmd[@]}" >> "$run/eval.log" 2>&1 || { echo "  FAILED (see $run/eval.log)"; fail=1; }
  done
done
exit $fail
