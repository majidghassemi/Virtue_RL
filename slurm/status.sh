#!/bin/bash
# Progress and projected finish for every run. Run on a login node:
#
#   bash slurm/status.sh
#
# Reads what is actually on disk: each run's own config.json for its episode
# target, and log.csv for progress. Deliberately does NOT depend on $GRID --
# forgetting to set it would otherwise report every run against the wrong target.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh

echo "── queue ──────────────────────────────────────────────"
squeue -u "$USER" -o "%.12i %.12j %.3t %.11M %.11l %R" 2>/dev/null || echo "  (squeue unavailable)"

echo ""
echo "── runs ───────────────────────────────────────────────"
shopt -s nullglob
dirs=(sociapl/runs/*/)
if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "  none yet (sociapl/runs is empty)"
    exit 0
fi

printf '  %-16s %9s %9s %6s %8s %10s %s\n' RUN EPISODES TARGET PCT EP/SEC ETA STATE
for d in "${dirs[@]}"; do
    name="$(basename "$d")"
    log="$d/log.csv"; cfg="$d/config.json"

    # Target from the run's own config.json -- authoritative, and correct even
    # if this shell has a different GRID than the one that launched the job.
    target=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['episodes'])" "$cfg" 2>/dev/null || echo 0)

    if [[ ! -f "$log" ]]; then
        printf '  %-16s %9s %9s %6s %8s %10s %s\n' "$name" 0 "${target:-?}" - - - "no batches yet"
        continue
    fi
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    awk -F, -v n="$name" -v target="$target" -v age="$age" '
        NR>1 && $1 ~ /^[0-9]+$/ { eps=$1; sec=$NF; rows++
            if (prev_sec != "" && sec < prev_sec) { base_e=prev_eps; base_s=0 }  # pass boundary
            prev_sec=sec; prev_eps=eps }
        END {
            if (rows == 0) { printf "  %-16s %9d %9s %6s %8s %10s %s\n", n, 0, target, "-", "-", "-", "no batches yet"; exit }
            d_e = eps - (base_e ? base_e : 0); d_s = sec - base_s
            rate = (d_s > 0) ? d_e / d_s : 0
            pct = (target > 0) ? 100 * eps / target : 0
            if (target > 0 && eps >= target)      { etas = "done"; state = "complete" }
            else if (rate > 0 && target > 0)      { etas = sprintf("%.0fh", (target - eps) / rate / 3600) }
            else                                    etas = "?"
            if (state == "") state = (age < 900) ? "running" : sprintf("STALE %dm", age/60)
            printf "  %-16s %9d %9s %5.1f%% %8.2f %10s %s\n", n, eps, target, pct, rate, etas, state
        }' "$log"
done

echo ""
echo "  TARGET is each run's own --episodes, read from its config.json."
echo "  EP/SEC and ETA cover the CURRENT pass only ('sec' restarts each pass)."
echo "  STALE = log.csv untouched for 15+ min. Walltime per pass: $WALLTIME."
