#!/bin/bash
#SBATCH --account=def-YOURPI   # overridden by cc_sbatch.sh / submit.sh from ../.env
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-06:00
#SBATCH --output=slurm_evals_%j.out

# Re-evaluate runs whose eval.json was produced from an early snapshot, on the FINAL ckpt.pt.
# The three evaluations run in parallel on CPU (batch-1 inference gains nothing from a GPU).
# Runs from before the environment amendment have no new keys in config.json, so they are
# evaluated in their original environment. Also works as a plain `bash rerun_evals.sh`.
if command -v module >/dev/null 2>&1; then module load StdEnv/2023 python/3.11; fi
VENV=${VENV:-$HOME/venvs/virtue_rl}; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export OMP_NUM_THREADS=1
RUNS=(e_r0_solo_s1 e_r1_solo_s1 e_r2_solo_s0)
pids=()
for r in "${RUNS[@]}"; do
  ck=runs/$r/ckpt.pt
  if [ ! -f "$ck" ]; then echo "skip $r: no $ck"; continue; fi
  echo "== $r -> runs/$r/eval.json"
  python eval_ethics.py --ckpt "$ck" --episodes 100 --device cpu --out runs/$r/eval.json > runs/$r/eval.log 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done; exit $fail
