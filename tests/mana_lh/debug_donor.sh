#!/bin/bash
# Donor-stack checks for ~/mana-nccl-sp b94003d on Perlmutter (2026-10-08):
#   (1) where the donor stacks sit relative to libmana.so's writable segments,
#       and how large the upper half's static TLS is (glibc puts it inside a
#       stack passed with pthread_attr_setstack);
#   (2) whether the checkpoint survives (b94003d as is dies with
#       "libmana.so: undefined symbol: , version GLIBC_2.2.5").
# Runs once per MANA tree given in MANA_TREES.
#
#   salloc ... srun -N 1 -n 1 --overlap bash tests/mana_lh/debug_donor.sh

set -uo pipefail
: "${MANA_TREES:=$SCRATCH/mana-nccl-sp $SCRATCH/mana-nccl-sp-B}"
: "${BENCH:=$SCRATCH/nccl-ckpt-report/bin/nccl_bench_mpi}"
OUT=$SCRATCH/mana-nccl-sp-runs/donor-${SLURM_JOB_ID:?run inside a Slurm allocation}
RC=$HOME/.mana-slurm-$SLURM_JOB_ID.rc
module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 nccl/2.29.2-cu13 >/dev/null 2>&1
export DARSHAN_DISABLE=1 NCCL_RAS_ENABLE=0 NCCL_DEBUG=WARN
coord() { awk '$1 == "Host:" {h = $2} $1 == "Port:" {p = $2} END {print "-h " h " -p " p}' "$RC"; }
region() { # <pid> <addr>: the maps line containing addr, plus its neighbours
    awk -v a=$(($2)) '{split($1, r, "-"); b = strtonum("0x" r[1]); e = strtonum("0x" r[2]);
        line[NR] = $0; if (a >= b && a < e) hit = NR}
        END {if (hit) {print "      prev: " line[hit-1]; print "      HERE: " line[hit]; print "      next: " line[hit+1]}
             else print "      (unmapped)"}' /proc/$1/maps
}

for MANA in $MANA_TREES; do
    tag=$(basename "$MANA")
    D=$OUT/$tag
    mkdir -p "$D" && cd "$D" || continue
    rm -rf ckpt_rank_* dmtcp_restart_script*.sh
    "$MANA/bin/mana_coordinator" -p 0 >coord.log 2>&1
    # TASK_ENV (e.g. LD_BIND_NOW=1) goes to the tasks only: srun and salloc
    # themselves dlopen Slurm plugins that fail under LD_BIND_NOW.
    srun --export=ALL${TASK_ENV:+,$TASK_ENV} -N 1 -n 2 --ntasks-per-node=2 --gpus-per-node=2 -c 64 --cpu-bind=cores \
        "$MANA/bin/mana_launch" --cuda "$BENCH" -b allreduce -f "$D/id.bin" -p 200 \
        -m 4194304:4194304 -T 60 >launch.log 2>&1 &
    pid=$!
    for _ in $(seq 1 300); do grep -q "^# Size" launch.log 2>/dev/null && break; sleep 0.5; done
    sleep 3
    p=$(pgrep -u "$USER" -f "bin/lower-half.*nccl_bench_mpi" | head -1)
    {
        echo "######## $tag  (libmana $(md5sum "$MANA/lib/dmtcp/libmana.so" | cut -c1-8))  rank pid $p"
        echo "-- libmana.so mappings:"
        grep "libmana.so" /proc/$p/maps | sed 's/^/   /'
        echo "-- upper-half static TLS (PT_TLS MemSiz of each upper-half module):"
        tot=0
        for f in $(awk '$6 ~ /^\// && strtonum("0x" substr($1, 1, index($1, "-") - 1)) < 0x7f0000000000 {print $6}' /proc/$p/maps | sort -u); do
            m=$(readelf -lW "$f" 2>/dev/null | awk '$1 == "TLS" {print strtonum($6)}')
            [ -n "$m" ] && [ "$m" -gt 0 ] && { echo "   $m  $f"; tot=$((tot + m)); }
        done
        echo "   total: $tot bytes"
        echo "-- threads: tid name sp fs_base, and the region holding the stack"
        gdb -p "$p" -batch -ex 'thread apply all p/x $sp' -ex 'thread apply all p/x $fs_base' 2>/dev/null |
            grep -E '^Thread|^\$' >gdb_regs.txt
        awk '/^Thread/ {match($0, /LWP [0-9]+/); t = substr($0, RSTART + 4, RLENGTH - 4);
                        match($0, /"[^"]*"/); n[t] = substr($0, RSTART, RLENGTH); cur = t; next}
             /^\$/ {v = $NF; if (!(cur in sp)) sp[cur] = v; else fs[cur] = v}
             END {for (t in sp) print t, n[t], sp[t], fs[t]}' gdb_regs.txt | sort -n | while read -r t n s f; do
            echo "   $t $n sp=$s fs=$f"
            region "$p" "$s"
        done
    } >live.txt 2>&1
    rm -f dmtcp_restart_script*.sh
    "$MANA/bin/dmtcp_command" $(coord) -c
    ok=0
    for _ in $(seq 1 120); do
        compgen -G 'dmtcp_restart_script*.sh' >/dev/null && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 0.5
    done
    echo "== $tag: checkpoint $([ $ok = 1 ] && echo COMPLETED || echo FAILED)" | tee -a live.txt
    grep -m2 -E "symbol lookup error|Segmentation|JASSERT" launch.log | tee -a live.txt
    "$MANA/bin/dmtcp_command" $(coord) --kill >/dev/null 2>&1
    "$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
done
echo "results: $OUT/*/live.txt"
