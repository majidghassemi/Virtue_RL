#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0-12:00
#SBATCH --output=slurm_evals_%j.out

# Re-evaluate runs whose eval.json was produced from an early snapshot, on the FINAL ckpt.pt.
# Runs from before the environment amendment have no new keys in config.json, so they are
# evaluated in their original environment. Also works as a plain `bash rerun_evals.sh`.
RUNS=(e_r0_solo_s1 e_r1_solo_s1 e_r2_solo_s0)
for r in "${RUNS[@]}"; do
  ck=runs/$r/ckpt.pt
  if [ ! -f "$ck" ]; then echo "skip $r: no $ck"; continue; fi
  echo "== $r"
  python eval_ethics.py --ckpt "$ck" --episodes 100 --out runs/$r/eval.json
done
