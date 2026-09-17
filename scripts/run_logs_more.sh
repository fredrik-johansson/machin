#!/bin/sh
# Improve the 40- and 48-prime logarithm sets: alternate
#   stage b: extension of the many cheap elements (x >= FROM_B, tau <= TAU_B)
#   stage a: extension of the large elements      (x >= FROM_A, tau <= TAU_A)
# (each with adaptive combination rounds) until a whole cycle adds no new x,
# or MAXCYCLES / the time limit is reached.  Elements already extended
# are recorded in done<NP>.txt and never extended twice, also across
# restarts.  Restartable: rerun the script to continue.
#
# Needs machin_set built from the current machin_set.c
# (MACHIN_SET_EXTEND_DONE).
#
# Starting sets (the first one found is used):
#   NP=40: r40c_0.txt, np40_log_heuristic_x.txt, r40b_0.txt
#   NP=48: r48b_0.txt, r48a_0.txt
#   (or START=file for a single NP)

MS=${MS:-./machin_set}
export FLINT_NUM_THREADS=${FLINT_NUM_THREADS:-8}
NPS=${NPS:-"40 48"}
MAXCYCLES=${MAXCYCLES:-6}
ROUNDS=${ROUNDS:-50}
HOURS=${HOURS:-4}               # do not start a new stage after this
FROM_B=${FROM_B:-1e14}; TAU_B=${TAU_B:-2e6}
FROM_A=${FROM_A:-1e17}; TAU_A=${TAU_A:-1e9}
SRC="sieve, seeds from smaller sets, combination rounds and extension"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))

minx() {    # smallest selected x of a formula (from its last round line)
    grep "^# round" "$1" 2>/dev/null | tail -1 | sed 's/.*selected x from \([0-9]*\).*/\1/'
}
bigger() {  # is decimal $1 > decimal $2 ?
    [ -z "$2" ] && return 0
    [ -z "$1" ] && return 1
    [ ${#1} -gt ${#2} ] && return 0
    [ ${#1} -lt ${#2} ] && return 1
    [ "$1" \> "$2" ]
}
lines() { [ -s "$1" ] && wc -l < "$1" || echo 0; }

for NP in $NPS; do
    cur=cur${NP}_0.txt
    if [ ! -s $cur ]; then
        if [ -n "$START" ]; then cands=$START
        elif [ $NP = 40 ]; then cands="r40c_0.txt np40_log_heuristic_x.txt r40b_0.txt"
        else cands="r48b_0.txt r48a_0.txt"; fi
        for f in $cands; do [ -s $f ] && { cp $f $cur; break; }; done
        [ -s $cur ] || { echo "NP=$NP: no starting set found"; continue; }
    fi
    echo "== NP=$NP: starting from $(lines $cur) x"

    c=1
    stop=0
    while [ $c -le $MAXCYCLES ] && [ $stop = 0 ]; do
        # set size at the start of the cycle (kept across restarts)
        [ -s start${NP}_c$c.txt ] || lines $cur > start${NP}_c$c.txt
        before=$(cat start${NP}_c$c.txt)
        for st in b a; do
            tab=form${NP}_c${c}${st}.py
            [ -s $tab ] && grep -q "verification: ok" $tab && continue   # done earlier
            if [ $(date +%s) -ge $DEADLINE ]; then
                echo "   time limit reached (rerun to continue)"; stop=1; break
            fi
            if [ $st = b ]; then FROM=$FROM_B; TAU=$TAU_B; else FROM=$FROM_A; TAU=$TAU_A; fi
            s=$(date +%s)
            MACHIN_SET_SAVE=next${NP}.txt MACHIN_SET_SOURCE="$SRC" \
            MACHIN_SET_EXTEND=10000000 MACHIN_SET_EXTEND_MAXTAU=$TAU \
            MACHIN_SET_EXTEND_FROM=$FROM MACHIN_SET_EXTEND_DONE=done${NP}.txt \
                $MS 0 $NP load $cur $ROUNDS 0 1 > $tab.tmp 2> log${NP}_c${c}${st}.txt
            if grep -q "verification: ok" $tab.tmp && [ -s next${NP}.txt ]; then
                mv $tab.tmp $tab
                mv next${NP}.txt $cur
            else
                echo "   cycle $c stage $st FAILED, see log${NP}_c${c}${st}.txt"; exit 1
            fi
            m=$(minx $tab)
            echo "   cycle $c stage $st: $(( $(date +%s) - s )) s, $(lines $cur) x, min x $m"
            if bigger "$m" "$(minx best${NP}_0.py)"; then cp $tab best${NP}_0.py; fi
        done
        [ $stop = 1 ] && break
        if [ $(lines $cur) -eq $before ]; then echo "   no new x in cycle $c: done"; break; fi
        c=$((c + 1))
    done
    echo "== NP=$NP: best formula best${NP}_0.py, min x $(minx best${NP}_0.py)"
done
