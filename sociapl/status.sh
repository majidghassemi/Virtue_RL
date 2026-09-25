#!/bin/bash
# Progress and projected finish for every run. Run on a login node:
#
#   bash status.sh                 # RUN_ROOT=runs/v2
#   RUN_ROOT=runs/tune bash status.sh
#
# Reads what is on disk: each run's own config.json for its episode target and
# log.csv for progress. Deliberately does NOT infer the target from a grid script --
# doing that once reported every run against the wrong target.
set -euo pipefail
cd "$(dirname "$0")"
source cc_env.sh
RUN_ROOT=${RUN_ROOT:-runs/v2}

echo "── queue ──────────────────────────────────────────────"
squeue -u "$USER" -o "%.12i %.12j %.3t %.11M %.11l %R" 2>/dev/null || echo "  (squeue unavailable)"

echo
echo "── runs under $RUN_ROOT ───────────────────────────────"
shopt -s nullglob
dirs=("$RUN_ROOT"/*/)
if [ ${#dirs[@]} -eq 0 ]; then echo "  none yet"; exit 0; fi

printf '  %-28s %9s %9s %6s %8s %9s %s\n' RUN EPISODES TARGET PCT EP/SEC ETA STATE
for d in "${dirs[@]}"; do
  name=$(basename "$d"); log="$d/log.csv"; cfg="$d/config.json"
  target=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['episodes'])" "$cfg" 2>/dev/null || echo 0)
  if [ ! -f "$log" ]; then
    printf '  %-28s %9s %9s %6s %8s %9s %s\n' "$name" 0 "$target" - - - "no batches yet"; continue
  fi
  age=$(( $(date +%s) - $(stat -c %Y "$log") ))
  awk -F, -v n="$name" -v target="$target" -v age="$age" '
    NR==1 { for (i=1;i<=NF;i++) { if ($i=="episodes") ec=i; if ($i=="sec") sc=i } ; next }
    $ec ~ /^[0-9]+$/ { eps=$ec; sec=$sc; rows++
      if (psec != "" && sec < psec) { be=peps; bs=0 }     # pass boundary: sec restarts
      psec=sec; peps=eps }
    END {
      if (rows==0) { printf "  %-28s %9d %9s %6s %8s %9s %s\n", n, 0, target, "-","-","-","no batches yet"; exit }
      de = eps - (be ? be : 0); ds = sec - bs
      rate = (ds > 0) ? de/ds : 0
      pct  = (target > 0) ? 100*eps/target : 0
      if (target > 0 && eps >= target) { eta="done"; state="complete" }
      else if (rate > 0 && target > 0) eta=sprintf("%.0fh",(target-eps)/rate/3600)
      else eta="?"
      if (state=="") state = (age < 1800) ? "running" : sprintf("STALE %dm", age/60)
      printf "  %-28s %9d %9s %5.1f%% %8.2f %9s %s\n", n, eps, target, pct, rate, eta, state
    }' "$log"
done
echo
echo "  TARGET is each run's own --episodes from config.json."
echo "  EP/SEC and ETA cover the CURRENT pass ('sec' restarts each pass)."
echo "  STALE = log.csv untouched for 30+ min."
