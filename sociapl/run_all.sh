#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --array=0-69
#SBATCH --cpus-per-task=8
##SBATCH --gpus-per-node=1   # uncomment to train on a GPU (--device auto picks it up)
#SBATCH --mem=8G
#SBATCH --time=6-23:00
#SBATCH --output=slurm_%A_%a.out

# Full experiment grid, 5 seeds per condition, 800k episodes each (70 jobs):
#   R0 {virt, solo, short} x detour {zero, small, large}   45 jobs  $RUN_ROOT/e_<cond>_d<detour>_s<seed>
#   R2 solo, dense lambda in LAMBDAS                        15 jobs  $RUN_ROOT/e_r2_solo_l<lambda>_s<seed>
#   R1 {solo, virt} (delayed), frozen detour level          10 jobs  $RUN_ROOT/e_<cond>_s<seed>
# If you change the arrays below, update --array to 0..(jobs-1); the script prints the total.
#
# Requires the frozen environment (README "Environment tuning"): ENV_CONFIG, default env_frozen.json.
# Each SLURM pass runs until walltime; ckpt.pt auto-resumes (weights, optimizer, episode
# count, and it refuses to resume under a different environment), so chain ~4 passes:
#
#   jid=$(sbatch --parsable run_all.sh)
#   for i in 1 2 3; do jid=$(sbatch --parsable --dependency=afterany:$jid run_all.sh); done
#
# Finished runs exit immediately with "nothing to do", so over-chaining is harmless.
# Resume a single dead index K: sbatch --array=K run_all.sh
# Shakedown before the real thing: sbatch --array=0 --time=0-00:30 run_all.sh

SEEDS=(0 1 2 3 4)
R0_CONDS=(r0_virt r0_solo r0_short)
DETOURS=(zero small large)
LAMBDAS=(0.1 0.25 0.5)     # R2 dense penalty, below the per-goal reward of 1
R1_CONDS=(r1_solo r1_virt)
ENV_CONFIG=${ENV_CONFIG:-env_frozen.json}
RUN_ROOT=${RUN_ROOT:-runs/v2}   # post-amendment runs; keeps old runs/e_* untouched

JOBS=()
for c in "${R0_CONDS[@]}"; do for d in "${DETOURS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$c $d - $s"); done; done; done
for l in "${LAMBDAS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("r2_solo - $l $s"); done; done
for c in "${R1_CONDS[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("$c - - $s"); done; done
echo "grid has ${#JOBS[@]} jobs"

if [ ! -f "$ENV_CONFIG" ]; then
  echo "missing $ENV_CONFIG: tune and freeze the environment first (python summarize_tuning.py --freeze ...)"; exit 1
fi
if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#JOBS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID outside 0..$(( ${#JOBS[@]} - 1 ))"; exit 1
fi
read C D L S <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

module load StdEnv/2023 python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch numpy tqdm || pip install torch numpy tqdm
pip install pyglet gym==0.26.2 gym-minigrid numba
pip install -e ../marlgrid --no-deps

case $C in
  r0_virt)  ARGS="--mode social --virtuous 1 --harm_delivery none";;
  r0_solo)  ARGS="--mode solo                --harm_delivery none";;
  r0_short) ARGS="--mode social --virtuous 0 --harm_delivery none";;
  r2_solo)  ARGS="--mode solo                --harm_delivery dense --harm_lambda $L";;
  r1_solo)  ARGS="--mode solo                --harm_delivery delayed";;
  r1_virt)  ARGS="--mode social --virtuous 1 --harm_delivery delayed";;
esac
OUT=$RUN_ROOT/e_${C}_s${S}
if [ "$D" != "-" ]; then ARGS="$ARGS --harm_detour $D"; OUT=$RUN_ROOT/e_${C}_d${D}_s${S}; fi
if [ "$L" != "-" ]; then OUT=$RUN_ROOT/e_${C}_l${L}_s${S}; fi

python train_ethics.py $ARGS --env_config $ENV_CONFIG --episodes 800000 --n_envs 16 --batch_episodes 128 \
  --threads 8 --seed $S --snapshot_every 40000 --out $OUT
