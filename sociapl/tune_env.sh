#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --array=0-35
#SBATCH --cpus-per-task=8
##SBATCH --gpus-per-node=1   # uncomment to train on a GPU (--device auto picks it up)
#SBATCH --mem=8G
#SBATCH --time=2-00:00
#SBATCH --output=slurm_tune_%A_%a.out

# Environment tuning sweep (run BEFORE any full run; see README "Environment tuning").
#   - R0 only, solo and virtuous-teacher conditions, harm-free route never longer (--harm_detour zero).
#   - --hide_harm 1: harm columns are blanked in log.csv/stdout; tune on task metrics only.
#   - grid: wrong-goal penalty x number of goals x view size = 3 x 3 x 2 cells, x 2 conditions, 1 seed.
# TUNE_EPISODES must be long enough for R0-virt to cross return 1 under the current env;
# 100k is a placeholder -- set it from your own earlier R0-virt curves:
#   sbatch --export=ALL,TUNE_EPISODES=150000 tune_env.sh
# Then: python summarize_tuning.py runs/tune

PENALTIES=(-1.5 -3 -5)
GOALS=(3 4 5)
VIEWS=(5 7)
CONDS=(r0_solo r0_virt)
SEED=0
EPISODES=${TUNE_EPISODES:-100000}

i=$SLURM_ARRAY_TASK_ID
C=${CONDS[$((i % 2))]};              i=$((i / 2))
V=${VIEWS[$((i % 2))]};              i=$((i / 2))
G=${GOALS[$((i % 3))]};              i=$((i / 3))
P=${PENALTIES[$((i % 3))]}

module load StdEnv/2023 python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch numpy tqdm || pip install torch numpy tqdm
pip install pyglet gym==0.26.2 gym-minigrid numba
pip install -e ../marlgrid --no-deps

case $C in
  r0_solo) ARGS="--mode solo";;
  r0_virt) ARGS="--mode social --virtuous 1";;
esac

python train_ethics.py $ARGS --harm_delivery none --harm_detour zero --hide_harm 1 \
  --penalty $P --n_goals $G --view_size $V \
  --episodes $EPISODES --n_envs 16 --batch_episodes 128 --threads 8 --seed $SEED \
  --out runs/tune/p${P}_g${G}_v${V}_${C}_s${SEED}
