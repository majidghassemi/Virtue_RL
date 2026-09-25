#!/bin/bash
# sbatch a ONE-OFF job with the account and mail settings from ../.env, so the
# placeholder #SBATCH --account line in the script never has to be edited.
#
#   bash cc_sbatch.sh smoke_test.sh
#   bash cc_sbatch.sh rerun_evals.sh
#   bash cc_sbatch.sh --time=1:00:00 smoke_test.sh     # extra sbatch flags pass through
#
# Grid scripts do NOT need this -- submit.sh injects the same flags and adds
# --array and the resume chain. Use this only for scripts you would otherwise
# `sbatch` directly.
set -euo pipefail
cd "$(dirname "$0")"
source cc_env.sh
mapfile -t CFG < <(cc_sbatch_opts)
echo "sbatch ${CFG[*]} $*"
exec sbatch "${CFG[@]}" "$@"
