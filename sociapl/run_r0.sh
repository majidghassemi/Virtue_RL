#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
##SBATCH --gpus-per-node=1   # uncomment to train on a GPU (--device auto picks it up)
#SBATCH --mem=8G
#SBATCH --time=6-23:00
#SBATCH --output=slurm_%A_%a.out

# R0 kill experiment only: {virt, solo, short} x 5 seeds, frozen environment (incl. its detour level).
CONDS=(virt virt virt virt virt solo solo solo solo solo short short short short short)
SEEDS=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
C=${CONDS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ENV_CONFIG=${ENV_CONFIG:-env_frozen.json}
RUN_ROOT=${RUN_ROOT:-runs/v2}   # post-amendment runs; keeps old runs/e_* untouched
if [ ! -f "$ENV_CONFIG" ]; then
  echo "missing $ENV_CONFIG: tune and freeze the environment first (python summarize_tuning.py --freeze ...)"; exit 1
fi

module load StdEnv/2023 python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch numpy tqdm || pip install torch numpy tqdm
pip install pyglet gym==0.26.2 gym-minigrid numba
pip install -e ../marlgrid --no-deps

case $C in
  virt)  ARGS="--mode social --virtuous 1";;
  short) ARGS="--mode social --virtuous 0";;
  solo)  ARGS="--mode solo";;
esac

# ckpt.pt in the out dir auto-resumes (weights, optimizer, episode count); resubmits are safe
python train_ethics.py $ARGS --harm_delivery none --env_config $ENV_CONFIG --episodes 200000 \
  --n_envs 16 --batch_episodes 128 --threads 8 --seed $S \
  --snapshot_every 20000 --out $RUN_ROOT/e_r0_${C}_s${S}
