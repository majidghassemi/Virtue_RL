"""Summarise the environment-tuning sweep on TASK metrics only (harm columns are never read).

  python summarize_tuning.py runs/tune                      # table + which cells pass
  python summarize_tuning.py runs/tune --freeze runs/tune/p-3_g4_v5_r0_virt_s0 --out env_frozen.json

Criterion (per env cell = penalty x goals x view x detour x layouts): the solo learner's final
return stays at or below --solo_max (plateau near 1) in every seed, AND the virtuous-teacher
learner's final return is at least --virt_min in a majority of seeds. "Final" = mean
learner_return over the last --last batches. Thresholds are defaults, not results: set them
before looking at the table.

--freeze copies the environment keys of one run's config.json into a JSON for
train_ethics.py --env_config, and prints a dated amendment block for HYPOTHESES.md.
"""
import argparse, csv, datetime, glob, json, os
from collections import defaultdict
import numpy as np
from ethics import ENV_KEYS

TASK_COLS = ("episodes", "learner_return")  # the only log columns this script reads


def final_return(log_path, last):
    with open(log_path, newline="") as f:
        rows = [(int(r["episodes"]), float(r["learner_return"])) for r in csv.DictReader(f)]
    if not rows:
        return None, 0
    return float(np.mean([r for _, r in rows[-last:]])), rows[-1][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="runs/tune")
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--solo_max", type=float, default=1.5)
    p.add_argument("--virt_min", type=float, default=2.0)
    p.add_argument("--freeze", default=None, help="run dir whose environment to freeze")
    p.add_argument("--out", default="env_frozen.json")
    a = p.parse_args()

    if a.freeze:
        with open(os.path.join(a.freeze, "config.json")) as f:
            cfg = json.load(f)
        env = {k: cfg[k] for k in ENV_KEYS if k in cfg}
        with open(a.out, "w") as f:
            json.dump(env, f, indent=2)
        print(f"wrote {a.out}\n")
        print(f"### Amendment {datetime.date.today().isoformat()}: frozen environment\n")
        print(f"Chosen from the tuning sweep in `{a.root}` (R0 solo and R0 virt only, task metrics only,")
        print(f"harm columns hidden during tuning). Criterion: solo final return <= {a.solo_max} in all seeds,")
        print(f"virt final return >= {a.virt_min} in a majority of seeds (last {a.last} batches).")
        print(f"Frozen before any full run; all full runs use `--env_config {a.out}`:\n")
        print("```json\n" + json.dumps(env, indent=2) + "\n```")
        return

    cells = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(os.path.join(a.root, "*"))):
        cp, lp = os.path.join(d, "config.json"), os.path.join(d, "log.csv")
        if not (os.path.exists(cp) and os.path.exists(lp)):
            continue
        with open(cp) as f:
            cfg = json.load(f)
        cond = "solo" if cfg["mode"] == "solo" else ("virt" if cfg.get("virtuous", 1) else "short")
        key = tuple((k, json.dumps(cfg.get(k))) for k in ENV_KEYS)
        ret, eps = final_return(lp, a.last)
        if ret is not None:
            cells[key][cond].append((ret, eps, d))

    print(f"{'penalty':>7} {'goals':>5} {'view':>4}  {'solo final (per seed)':<24} {'virt final (per seed)':<24} {'episodes':>9}  pass")
    for key, conds in cells.items():
        k = {n: json.loads(v) for n, v in key}
        solo = [r for r, _, _ in conds.get("solo", [])]
        virt = [r for r, _, _ in conds.get("virt", [])]
        eps = min(e for c in conds.values() for _, e, _ in c)
        ok = bool(solo and virt and all(r <= a.solo_max for r in solo)
                  and sum(r >= a.virt_min for r in virt) > len(virt) / 2)
        fmt = lambda xs: " ".join(f"{x:.2f}" for x in xs) or "-"
        print(f"{k['penalty']:>7} {k['n_goals']:>5} {k['view_size']:>4}  {fmt(solo):<24} {fmt(virt):<24} {eps:>9}  {'YES' if ok else ''}")


if __name__ == "__main__":
    main()
