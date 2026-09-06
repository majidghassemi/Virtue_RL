#!/bin/bash
# Build the venv. Run ONCE, on a cluster LOGIN NODE, from the project directory:
#
#   bash slurm/setup.sh
#
# Must be a login node: compute nodes have no outbound internet, so pip cannot
# reach PyPI there. Every job script just activates what this builds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh

[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: run on a login node, not in a job (no internet there)." >&2; exit 1; }

module load $MODULES
echo "python: $(python --version)"

if [[ -d venv && -z "${FORCE:-}" ]]; then
    echo "venv/ exists — reusing it (FORCE=1 to rebuild)."
    source venv/bin/activate
else
    rm -rf venv
    virtualenv --no-download venv
    source venv/bin/activate
    pip install --no-index --upgrade pip

    # Heavy compiled packages from DRAC's wheelhouse; the rest from PyPI.
    pip install --no-index torch numpy tqdm
    # gym 0.26.2 has legacy metadata that newer pip can reject.
    pip install -r requirements.txt || {
        echo "retrying without build isolation..."
        pip install "setuptools<67" wheel
        pip install --no-build-isolation -r requirements.txt
    }
    # --no-deps: marlgrid/setup.py has unpinned gym pins that would undo the above.
    pip install -e ./marlgrid --no-deps
fi

mkdir -p logs sociapl/runs

echo ""
echo "venv ready. Next:  bash slurm/test.sh"
