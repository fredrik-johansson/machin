#!/bin/sh
# Heuristic search for Machin-type sets with NP = 48 (logarithms and
# arctangents), seeded with the x lists of earlier runs.  About an hour
# on 8 threads.  Restartable: finished steps are skipped (delete a
# step's output to redo it).
#
# Seed files used if present (any smooth x over fewer primes is smooth
# over the first 48 too; missing files are skipped):
#   logarithms:   np48_log_seed_x.txt (my earlier NP=48 set),
#                 np40_log_heuristic_x.txt or r40c_0.txt / r40b_0.txt,
#                 np32_0.txt (optimal 32-prime list), s40_0.txt
#   arctangents:  r40b_1.txt / r40_1.txt, np32_1.txt, s40_1.txt

NP=${NP:-48}
LOGSIEVE=${LOGSIEVE:-1e12}      # ~2-3 min on 8 threads
ATSIEVE=${ATSIEVE:-1e13}        # ~15 min on 8 threads; 5e12 if short on time
MS=${MS:-./machin_set}
export FLINT_NUM_THREADS=${FLINT_NUM_THREADS:-8}
SRC="sieve, seeds from smaller sets, combination rounds and extension"

t() { s=$(date +%s); "$@"; echo "   [$(( $(date +%s) - s )) s] $*" >&2; }
best() { grep "^/\* round" "$1" | tail -1 | sed 's/.*selected x from \([0-9]*\).*/\1/'; }

cat_existing() {   # cat the files that exist
    for f in "$@"; do [ -s "$f" ] && cat "$f"; done
}

# ---- 1. sieves ----------------------------------------------------------
[ -s s${NP}_0.txt ] || MACHIN_SET_SAVE=s${NP}_0.txt t $MS 0 $NP $LOGSIEVE > /dev/null
[ -s s${NP}_1.txt ] || MACHIN_SET_SAVE=s${NP}_1.txt t $MS 1 $NP $ATSIEVE > /dev/null

# ---- 2. seeds -----------------------------------------------------------
[ -s seed${NP}_0.txt ] || cat_existing s${NP}_0.txt np48_log_seed_x.txt \
    np40_log_heuristic_x.txt r40c_0.txt r40b_0.txt s40_0.txt np32_0.txt > seed${NP}_0.txt
[ -s seed${NP}_1.txt ] || cat_existing s${NP}_1.txt r40b_1.txt r40_1.txt \
    s40_1.txt np32_1.txt > seed${NP}_1.txt

# ---- 3. logarithms --------------------------------------------------------
# a) rounds + extension of every element >= 1e17 (~7.5e9 tests, ~6 min)
[ -s tab${NP}a_0.c ] || MACHIN_SET_SAVE=r${NP}a_0.txt MACHIN_SET_SOURCE="$SRC" \
    MACHIN_SET_EXTEND=1000000 MACHIN_SET_EXTEND_MAXTAU=1e9 MACHIN_SET_EXTEND_FROM=1e17 \
    t $MS 0 $NP load seed${NP}_0.txt 10 0 1 > tab${NP}a_0.c 2> log${NP}a_0.txt
# b) lower threshold, cheap elements only (small tau)
[ -s tab${NP}b_0.c ] || MACHIN_SET_SAVE=r${NP}b_0.txt MACHIN_SET_SOURCE="$SRC" \
    MACHIN_SET_EXTEND=1000000 MACHIN_SET_EXTEND_MAXTAU=2e6 MACHIN_SET_EXTEND_FROM=1e14 \
    t $MS 0 $NP load r${NP}a_0.txt 10 0 1 > tab${NP}b_0.c 2> log${NP}b_0.txt

# ---- 4. arctangents -------------------------------------------------------
# small x are cheap to extend here (small tau), so extend almost everything
[ -s tab${NP}a_1.c ] || MACHIN_SET_SAVE=r${NP}a_1.txt MACHIN_SET_SOURCE="$SRC" \
    MACHIN_SET_EXTEND=1000000 MACHIN_SET_EXTEND_MAXTAU=1e9 MACHIN_SET_EXTEND_FROM=1e6 \
    t $MS 1 $NP load seed${NP}_1.txt 10 0 1 > tab${NP}a_1.c 2> log${NP}a_1.txt

# ---- summary --------------------------------------------------------------
for f in tab${NP}a_0.c tab${NP}b_0.c tab${NP}a_1.c; do
    [ -s $f ] && echo "$f: min x $(best $f), $(grep -c 'verification: ok' $f) verified"
done
