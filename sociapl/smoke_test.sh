#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-00:30
#SBATCH --output=slurm_smoke_%j.out

# End-to-end smoke test of every entry point with tiny episode counts (~5-10 min).
# Locally:              bash smoke_test.sh
# On Compute Canada:    sbatch smoke_test.sh        (checks the GPU path on a real GPU node)
# Uses the real job scripts (tune_env.sh, run_all.sh) through cc_common.sh, 2 runs packed
# per task. Everything is written to a scratch dir; exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if command -v module >/dev/null 2>&1; then module load StdEnv/2023 python/3.11; fi
VENV=${VENV:-$HOME/venvs/virtue_rl}; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PYTHONWARNINGS=ignore
T=${SMOKE_DIR:-${SLURM_TMPDIR:-$(mktemp -d)}/smoke}
rm -rf "$T"; mkdir -p "$T"
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
step() { echo; echo "=== $*"; }
q() { "$@" 2>&1 | grep -v "Gym\|gym\|fn()\|Users of\|migration" || true; }

step "environment"
python - <<'PY'
import torch, numpy, sys
from model import get_device
print("python", sys.version.split()[0], "| torch", torch.__version__, "| numpy", numpy.__version__)
print("cuda available:", torch.cuda.is_available(), "| device auto ->", get_device("auto"))
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

step "parallel env stepping == serial (identical rollouts)"
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from ethics import EthicsWorker
from vecenv import make_vec
from model import SociAPLNet
from ppo import collect
out = []
for n in (0, 3):
    envs = make_vec(EthicsWorker, [(("social", 2), dict(seed=i)) for i in range(6)], n)
    torch.manual_seed(0); net = SociAPLNet("pred")
    obs = envs.reset(); h, c = net.init_state(6); stats = []
    data, *_ = collect(net, envs, 6, obs, h, c, stats); envs.close()
    out.append({k: v.clone() for k, v in data.items()})
assert all(torch.equal(out[0][k], out[1][k]) for k in out[0]), "subprocess rollout differs from serial"
print("ok: identical rollouts with 0 and 3 env processes")
PY

step "train.py (replication) + evaluate.py"
q python train.py --mode social --aux pred --episodes 16 --batch_episodes 16 --n_envs 8 --n_procs 2 --out "$T/rep" | tail -2
q python evaluate.py --ckpt "$T/rep/ckpt.pt" --episodes 1 | tail -4

step "tune_env.sh: tasks 0 (2 runs packed on one task), harm hidden"
export TUNE_ROOT="$T/tune" SLURM_CPUS_PER_TASK=$CPUS
SMALL="--batch_episodes 8 --n_envs 4"
RUNS_PER_JOB=2 EXTRA_ARGS="--episodes 8 $SMALL" \
SLURM_ARRAY_TASK_ID=0 bash tune_env.sh
for d in "$T"/tune/*; do tail -1 "$d/log.csv"; done
python - "$T/tune" <<'PY'
import csv, glob, sys
for f in glob.glob(sys.argv[1] + "/*/log.csv"):
    for r in csv.DictReader(open(f)):
        assert r["learner_harm"] == r["harm_per_100_moves"] == r["bystander_return"] == "", f
print("ok: harm columns blank in tuning logs")
PY
q python summarize_tuning.py "$T/tune" --last 1
q python summarize_tuning.py "$T/tune" --freeze "$(ls -d "$T"/tune/*virt* | head -1)" --out "$T/env_frozen.json" | head -1

step "run_all.sh: tasks 0 and 69 (env subprocesses), then a resume pass to 16 episodes"
export ENV_CONFIG="$T/env_frozen.json" RUN_ROOT="$T/v2"
for i in 0 69; do RUNS_PER_JOB=1 EXTRA_ARGS="--episodes 8 $SMALL" SLURM_ARRAY_TASK_ID=$i bash run_all.sh; done
for i in 0 69; do RUNS_PER_JOB=1 EXTRA_ARGS="--episodes 16 $SMALL" SLURM_ARRAY_TASK_ID=$i bash run_all.sh; done
for d in "$T"/v2/*; do
  grep -q "resumed from .* at episode 8" "$d/stdout.log" || { echo "FAIL: no resume from episode 8 in $d"; exit 1; }
  [ "$(tail -1 "$d/log.csv" | cut -d, -f1)" -ge 16 ] || { echo "FAIL: $d did not reach 16 episodes"; exit 1; }
  echo "$(basename "$d"): $(tail -1 "$d/log.csv" | cut -d, -f1) episodes | $(grep -m1 '^device' "$d/stdout.log")"
done
python - "$T/v2" <<'PY'
import sys, torch
dev = "cuda" if torch.cuda.is_available() else None
if dev:
    import glob
    for f in glob.glob(sys.argv[1] + "/*/stdout.log"):
        assert "device: cuda" in open(f).read(), f"{f} did not train on the GPU"
    print("ok: runs trained on cuda")
PY

step "eval_ethics.py (seen / unseen_pos / unseen_struct)"
q python eval_ethics.py --ckpt "$(ls -d "$T"/v2/* | head -1)/ckpt.pt" --episodes 1 --out "$T/eval.json" | grep -E "virtue_gap|note"

step "bc.py"
q python bc.py --episodes 4 --epochs 1 --out "$T/bc" | tail -1

echo; echo "SMOKE TEST PASSED  (artifacts in $T)"
