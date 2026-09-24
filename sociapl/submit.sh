#!/bin/bash
# Submit a grid script as a SLURM array and chain resume passes behind it.
#   bash submit.sh run_all.sh            # 4 passes (each resumes from ckpt.pt)
#   bash submit.sh run_all.sh 6
#   RUNS_PER_JOB=2 bash submit.sh run_all.sh 4 --account=def-OTHER
# Pass N+1 starts when every task of pass N has ended (afterany); finished runs exit at once.
set -e
S=$1; P=${2:-4}; shift; [ $# -gt 0 ] && shift
N=$(bash "$S" --count)
echo "$S: $N array tasks (RUNS_PER_JOB=${RUNS_PER_JOB:-1}), $P passes"
jid=$(sbatch --parsable --array=0-$((N - 1)) "$@" "$S")
echo "pass 1: $jid"
for i in $(seq 2 "$P"); do
  jid=$(sbatch --parsable --array=0-$((N - 1)) --dependency=afterany:$jid "$@" "$S")
  echo "pass $i: $jid"
done
