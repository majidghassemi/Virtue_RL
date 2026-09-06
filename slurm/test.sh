#!/bin/bash
# Check the environment works end to end. Run on a LOGIN NODE after setup.sh:
#
#   bash slurm/test.sh
#
# Takes a couple of minutes. Exercises the same path the real jobs take:
# imports, environment, a short training run, checkpoint, resume, evaluation,
# and an offline wandb run. If this passes, run_all.sh will work.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh

[[ -f venv/bin/activate ]] || { echo "ERROR: no venv — run 'bash slurm/setup.sh' first." >&2; exit 1; }
module load $MODULES 2>/dev/null || true
source venv/bin/activate

OUT="sociapl/runs/_test"
rm -rf "$OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "── 1/5 imports and model ──────────────────────────────"
( cd sociapl && python -c "
import torch, numpy, wandb, envs, ethics, model, ppo, wandb_utils
n = model.SociAPLNet(aux='pred')
p = sum(x.numel() for x in n.parameters())
print(f'  torch {torch.__version__}  numpy {numpy.__version__}  wandb {wandb.__version__}')
print(f'  params {p:,}')
assert p == 668555, f'expected 668,555 params, got {p:,} — model does not match the paper'
o = ethics.EthicsWorker('social', seed=0).reset()
assert o.shape == (21, 21, 3), f'expected (21,21,3) obs, got {o.shape}'
print(f'  env obs {o.shape}')
" )

echo "── 2/5 training (128 episodes) ────────────────────────"
( cd sociapl && python train_ethics.py \
    --mode social --virtuous 1 --harm_delivery none \
    --episodes 128 --batch_episodes 64 --n_envs 4 --threads 4 \
    --seed 0 --out runs/_test --wandb --wandb_tags test )

echo "── 3/5 resume (to 192 episodes) ───────────────────────"
# Proves a walltime-killed pass continues instead of restarting, and that the
# second pass reuses the same wandb run id.
( cd sociapl && python train_ethics.py \
    --mode social --virtuous 1 --harm_delivery none \
    --episodes 192 --batch_episodes 64 --n_envs 4 --threads 4 \
    --seed 0 --out runs/_test --wandb --wandb_tags test )

echo "── 4/5 evaluation ─────────────────────────────────────"
( cd sociapl && python eval_ethics.py --ckpt runs/_test/ckpt.pt --episodes 2 \
    --out runs/_test/eval.json --wandb --wandb_dir runs/_test --wandb_step 192 )

echo "── 5/5 outputs ────────────────────────────────────────"
python - "$OUT" "$WANDB_DIR/wandb" <<'PY'
import csv, json, os, sys
out, wdir = sys.argv[1], sys.argv[2]
for f in ("log.csv", "ckpt.pt", "config.json", "wandb_id.txt", "eval.json"):
    p = os.path.join(out, f)
    assert os.path.exists(p), f"MISSING {p}"
    print(f"  {f:14s} {os.path.getsize(p):>9,} bytes")
rows = list(csv.DictReader(open(os.path.join(out, "log.csv"))))
eps = [int(r["episodes"]) for r in rows]
assert eps == sorted(eps) and len(set(eps)) == len(eps), f"episodes not monotonic: {eps}"
assert eps[-1] == 192, f"resume did not continue: last episode {eps[-1]}, expected 192"
print(f"  log.csv        {len(rows)} rows, episodes {eps}")
rid = open(os.path.join(out, "wandb_id.txt")).read().strip()
dirs = [d for d in os.listdir(wdir) if d.startswith("offline-run-") and d.endswith(rid) or rid in d]
assert dirs, f"no offline wandb run for id {rid} in {wdir}"
print(f"  wandb id       {rid} -> {len(dirs)} offline dir(s), all one run once synced")
gap = json.load(open(os.path.join(out, "eval.json")))["virtue_gap_seen"]
print(f"  virtue_gap_seen {gap:.3f}")
PY

rm -rf "$OUT"
echo ""
echo "PASS. Environment works. Next:  bash run_all.sh"
echo "  (the offline wandb runs from this test remain; sync or ignore them)"
