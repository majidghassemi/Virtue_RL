#!/bin/bash
# One-time environment build. Run on a Rorqual LOGIN NODE, from the repo root:
#
#   bash slurm/setup_env.sh
#
# This must not happen inside a job. DRAC compute nodes have no outbound
# internet, so `pip install` from PyPI only works here, on a login node. (The
# original sociapl/run_all.sh built its venv inside the job and would have died
# at setup on every array task.)
#
# Builds ~/Virtue_RL/venv, creates the $SCRATCH output tree, smoke-tests the
# imports, and offers to log in to wandb.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source slurm/env.sh

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: this is running inside SLURM job $SLURM_JOB_ID." >&2
    echo "  Compute nodes have no internet — run it on a login node instead." >&2
    exit 1
fi

echo "── Modules ────────────────────────────────────────────"
module load $MODULES
python --version

echo "── venv ───────────────────────────────────────────────"
if [[ -d venv && -z "${FORCE:-}" ]]; then
    echo "  venv/ already exists — reusing it (FORCE=1 to rebuild)."
else
    [[ -n "${FORCE:-}" ]] && rm -rf venv
    virtualenv --no-download venv
fi
source venv/bin/activate
pip install --no-index --upgrade pip

echo "── Dependencies ───────────────────────────────────────"
# Prefer DRAC's prebuilt wheelhouse for the heavy compiled packages; it is far
# faster and avoids compiling against the wrong toolchain.
pip install --no-index torch numpy tqdm

# The rest come from PyPI. gym==0.26.2 uses legacy setup.py metadata that newer
# pip/setuptools can reject; retry without build isolation before giving up.
if ! pip install -r requirements.txt; then
    echo ""
    echo "  Plain install failed — retrying gym without build isolation."
    echo "  (gym 0.26.2 has legacy metadata that modern pip can refuse.)"
    pip install "setuptools<67" wheel
    pip install --no-build-isolation -r requirements.txt
fi

# --no-deps is required: marlgrid/setup.py lists unpinned gym/gym-minigrid that
# would otherwise overwrite the pins above. This copy is patched — see
# marlgrid/PATCHES.md.
pip install -e ./marlgrid --no-deps

echo "── Output tree on \$SCRATCH ─────────────────────────────"
# $HOME has a file-COUNT quota and wandb writes thousands of small files, so
# runs/, logs/ and wandb/ all live on scratch. The symlinks let every script
# keep using plain relative paths.
mkdir -p "$RUNS_DIR" "$LOGS_DIR" "$SCRATCH_ROOT/wandb"
for pair in "sociapl/runs:$RUNS_DIR" "logs:$LOGS_DIR"; do
    link="${pair%%:*}"; target="${pair#*:}"
    if [[ -L "$link" ]]; then
        echo "  $link -> $(readlink "$link")"
    elif [[ -e "$link" ]]; then
        echo "  WARNING: $link exists and is not a symlink — leaving it alone." >&2
    else
        ln -s "$target" "$link"; echo "  $link -> $target"
    fi
done

echo "── Smoke test ─────────────────────────────────────────"
# Must run from sociapl/: the imports there are flat (`from ethics import ...`).
( cd sociapl && python -c "
import torch, numpy, wandb
import envs, ethics, model, ppo, wandb_utils
from model import SociAPLNet
n = SociAPLNet(aux='pred')
print('  torch      ', torch.__version__)
print('  numpy      ', numpy.__version__)
print('  wandb      ', wandb.__version__)
print('  params     ', f'{sum(p.numel() for p in n.parameters()):,}', '(expected 668,555)')
w = ethics.EthicsWorker('social', seed=0)
o = w.reset(); print('  env obs    ', o.shape, '(expected (21, 21, 3))')
" )

echo "── wandb ──────────────────────────────────────────────"
echo "  project: $WANDB_PROJECT${WANDB_ENTITY:+  entity: $WANDB_ENTITY}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "  API key found in .env — slurm/sync_wandb.sh is ready."
elif grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "  Logged in via ~/.netrc — slurm/sync_wandb.sh is ready."
else
    echo "  No credentials. Training does not need any (it runs offline), but"
    echo "  slurm/sync_wandb.sh does. Either set WANDB_API_KEY in .env locally"
    echo "  and re-run 'bash slurm/deploy.sh', or here:"
    echo ""
    echo "      source venv/bin/activate && wandb login"
    echo ""
fi

echo ""
echo "Setup complete. Next:"
echo "    GRID=pilot PASSES=1 WALLTIME=00:30:00 bash run_all.sh   # 30-min shakedown"
echo "See experiment.md for the full runbook."
