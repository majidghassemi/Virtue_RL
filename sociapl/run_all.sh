#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --array=0-17
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=6-23:00
#SBATCH --output=slurm_%A_%a.out

# Full experiment grid, 6 conditions x 3 seeds, 800k episodes each.
# Each SLURM pass runs until walltime; ckpt.pt auto-resumes (weights, optimizer,
# episode count), so chain ~4 passes per job to reach 800k:
#
#   jid=$(sbatch --parsable run_all.sh)
#   for i in 1 2 3; do jid=$(sbatch --parsable --dependency=afterany:$jid run_all.sh); done
#
# Finished runs exit immediately with "nothing to do", so over-chaining is harmless.
# Resume a single dead index K: sbatch --array=K run_all.sh
# Shakedown before the real thing: sbatch --array=0 --time=0-00:30 run_all.sh

CONDS=(r0_virt r0_virt r0_virt r0_solo r0_solo r0_solo r0_short r0_short r0_short \
       r2_solo r2_solo r2_solo r1_solo r1_solo r1_solo r1_virt r1_virt r1_virt)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)
C=${CONDS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

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
  r2_solo)  ARGS="--mode solo                --harm_delivery dense";;
  r1_solo)  ARGS="--mode solo                --harm_delivery delayed";;
  r1_virt)  ARGS="--mode social --virtuous 1 --harm_delivery delayed";;
esac

python train_ethics.py $ARGS --episodes 800000 --n_envs 16 --batch_episodes 128 \
  --threads 8 --seed $S --snapshot_every 40000 --out runs/e_${C}_s${S}