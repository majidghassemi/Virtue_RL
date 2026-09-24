#!/bin/bash
# One-time setup on a Compute Canada LOGIN node (compute nodes may have no internet).
# Builds a shared virtualenv that every job activates instead of pip-installing per job.
#   bash setup_cc.sh                 # default VENV=$HOME/venvs/virtue_rl
#   VENV=/project/def-YOURPI/$USER/virtue_rl bash setup_cc.sh
set -e
VENV=${VENV:-$HOME/venvs/virtue_rl}
module load StdEnv/2023 python/3.11
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy tqdm           # CC wheelhouse builds (GPU-enabled torch)
pip install pyglet gym==0.26.2 gym-minigrid==1.2.2 numba
pip install -e "$(dirname "$0")/../marlgrid" --no-deps
python -c "import torch, gym, marlgrid; print('torch', torch.__version__, 'built with CUDA', torch.version.cuda)"
echo "venv ready: $VENV   (jobs read VENV from the environment; default is the same path)"
