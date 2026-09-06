# The experiment grid, defined once.
#
# Sourced by BOTH run_all.sh (to size the array and check what is finished) and
# train_array.sh (to map $SLURM_ARRAY_TASK_ID back to a condition). Keeping it in
# one file is what stops the submitter and the job from ever disagreeing about
# which index means which run.
#
# Selected by $GRID:
#   pilot  3 conditions x 1 seed  @ 200k episodes   (3 tasks)
#   full   6 conditions x 3 seeds @ 800k episodes  (18 tasks)
#
# Extra knobs: SEEDS="0 1" overrides the seed list for either grid.

GRID="${GRID:-full}"

case "$GRID" in
  pilot)
    # sociapl/README.md: "run a 200k-episode pilot ... and check that ordering
    # (a)-(d) appears. If it does not, scaling to 1.5M will not rescue it."
    # These three conditions are exactly the stated kill experiment:
    # e_r0_virt vs e_r0_solo harm rate, with e_r0_short as the control.
    _CONDS=(r0_virt r0_solo r0_short)
    _SEEDS=(${SEEDS:-0})
    EPISODES="${EPISODES:-200000}"
    SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-20000}"   # METRICS.md's stated cadence
    ;;
  full)
    _CONDS=(r0_virt r0_solo r0_short r2_solo r1_solo r1_virt)
    _SEEDS=(${SEEDS:-0 1 2})
    EPISODES="${EPISODES:-800000}"
    SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-40000}"   # ~20 files x 2.7 MB per run
    ;;
  *)
    echo "ERROR: unknown GRID='$GRID' (expected 'pilot' or 'full')" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# Flatten conditions x seeds into the array index space. Index i therefore maps
# to GRID_CONDS[i] / GRID_SEEDS[i], and nothing else needs to know the shape.
GRID_CONDS=(); GRID_SEEDS=()
for _s in "${_SEEDS[@]}"; do
    for _c in "${_CONDS[@]}"; do
        GRID_CONDS+=("$_c"); GRID_SEEDS+=("$_s")
    done
done
GRID_N=${#GRID_CONDS[@]}
unset _s _c _CONDS _SEEDS

# Run directory name for a (condition, seed) pair. Mirrors the names used
# throughout METRICS.md and sociapl/README.md (e_r0_virt_s0, ...).
grid_run_name() { echo "e_${1}_s${2}"; }

# Condition -> train_ethics.py flags. Carried over verbatim from the original
# sociapl/run_all.sh; see train_ethics.py's docstring for the R0/R1/R2 mapping.
grid_cond_args() {
    case "$1" in
      r0_virt)  echo "--mode social --virtuous 1 --harm_delivery none";;
      r0_solo)  echo "--mode solo                --harm_delivery none";;
      r0_short) echo "--mode social --virtuous 0 --harm_delivery none";;
      r2_solo)  echo "--mode solo                --harm_delivery dense";;
      r1_solo)  echo "--mode solo                --harm_delivery delayed";;
      r1_virt)  echo "--mode social --virtuous 1 --harm_delivery delayed";;
      *) echo "ERROR: unknown condition '$1'" >&2; return 1;;
    esac
}
