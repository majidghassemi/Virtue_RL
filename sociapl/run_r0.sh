#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --array=0-8
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=6-23:00
#SBATCH --output=slurm_%A_%a.out

CONDS=(virt virt virt solo solo solo short short short)
SEEDS=(0 1 2 0 1 2 0 1 2)
C=${CONDS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

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
python train_ethics.py $ARGS --harm_delivery none --episodes 200000 \
  --n_envs 16 --batch_episodes 128 --threads 8 --seed $S \
  --snapshot_every 20000 --out runs/e_r0_${C}_s${S}