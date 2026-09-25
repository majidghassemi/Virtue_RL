#!/bin/bash
# Submit a grid script as a SLURM array and chain resume passes behind it.
#   bash submit.sh run_all.sh            # 4 passes (each resumes from ckpt.pt)
#   bash submit.sh run_all.sh 6
#   RUNS_PER_JOB=2 bash submit.sh run_all.sh 4 --account=def-OTHER
# Pass N+1 starts when every task of pass N has ended (afterany); finished runs exit at once.
set -e
cd "$(dirname "$0")"
source cc_env.sh
S=$1; P=${2:-4}; shift; [ $# -gt 0 ] && shift

# --account / --mail-user come from ../.env via cc_env.sh and override the
# placeholder #SBATCH lines in the grid scripts. Anything you pass after the pass
# count is appended, so it still wins over these.
mapfile -t CFG < <(cc_sbatch_opts)

N=$(bash "$S" --count)
echo "$S: $N array tasks (RUNS_PER_JOB=${RUNS_PER_JOB:-1}), $P passes, account $ACCOUNT"
jid=$(sbatch --parsable --array=0-$((N - 1)) "${CFG[@]}" "$@" "$S")
echo "pass 1: $jid"
for i in $(seq 2 "$P"); do
  jid=$(sbatch --parsable --array=0-$((N - 1)) --dependency=afterany:$jid "${CFG[@]}" "$@" "$S")
  echo "pass $i: $jid"
done
