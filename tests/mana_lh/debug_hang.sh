#!/bin/bash
# Reproduce the PRECHECKPOINT hang of NCCL-in-the-lower-half MANA and collect
# diagnostics while it hangs (2026-10-07: one rank busy-loops read() on an
# eventfd with a 4 MiB buffer, the other waits in DMTCP's local barrier).
#
#   salloc -A m4128 -q interactive -t 00:30:00 -N 1 -C gpu --gpus-per-node=4 \
#       srun -N 1 -n 1 --overlap bash tests/mana_lh/debug_hang.sh
#
# Output: $OUT/diag.txt (per rank: MANA events, threads, stuck thread's stack
# words mapped to library+offset, fd table) plus the case's logs.

set -uo pipefail
: "${MANA:=$SCRATCH/mana-nccl-sp}"
: "${BENCH:=$SCRATCH/nccl-ckpt-report/bin/nccl_bench_mpi}"
: "${HANG_S:=45}"   # no restart script after this long = hung
OUT=$MANA-runs/debug-${SLURM_JOB_ID:?run inside a Slurm allocation}
RC=$HOME/.mana-slurm-$SLURM_JOB_ID.rc
module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 nccl/2.29.2-cu13 >/dev/null 2>&1
export DARSHAN_DISABLE=1 NCCL_RAS_ENABLE=0 MANA_TIMING=1 NCCL_DEBUG=WARN
mkdir -p "$OUT" && cd "$OUT" || exit 1
rm -rf ckpt_rank_* dmtcp_restart_script*.sh
coord() { awk '$1 == "Host:" {h = $2} $1 == "Port:" {p = $2} END {print "-h " h " -p " p}' "$RC"; }

"$MANA/bin/mana_coordinator" -p 0 >coord.log 2>&1
srun -N 1 -n 2 --ntasks-per-node=2 --gpus-per-node=2 -c 64 --cpu-bind=cores \
    "$MANA/bin/mana_launch" --cuda "$BENCH" -b allreduce -f "$OUT/id.bin" -p 200 \
    -m 4194304:4194304 -T 120 >launch.log 2>&1 &
pid=$!
for _ in $(seq 1 600); do grep -q "^# Size" launch.log 2>/dev/null && break; sleep 0.5; done
sleep 5
echo "== checkpoint requested $(date +%T)" | tee diag.txt
"$MANA/bin/dmtcp_command" $(coord) -c
for _ in $(seq 1 $((HANG_S * 2))); do compgen -G 'dmtcp_restart_script*.sh' >/dev/null && break; sleep 0.5; done
if compgen -G 'dmtcp_restart_script*.sh' >/dev/null; then
    echo "== NO HANG: checkpoint completed $(date +%T)" | tee -a diag.txt
else
    echo "== HANG after ${HANG_S}s, collecting $(date +%T)" | tee -a diag.txt
    for p in $(pgrep -u "$USER" -f "bin/lower-half.*nccl_bench_mpi"); do
        {
            echo; echo "################ pid $p"
            echo "-- MANA events / plugin lines of this run:"
            grep -aE "EVENT_|Checkpoint overhead|teardown|consensus" launch.log | tail -n 12
            echo "-- threads (tid comm state wchan syscall):"
            for t in $(ls /proc/$p/task); do
                w=$(cat /proc/$p/task/$t/wchan 2>/dev/null)
                st=$(awk '{print $3}' /proc/$p/task/$t/stat)
                [ "$w" = futex_wait_queue ] && continue  # parked by DMTCP
                echo "   $t $(cat /proc/$p/task/$t/comm) $st $w $(cat /proc/$p/task/$t/syscall 2>/dev/null | cut -d' ' -f1-3)"
            done
            echo "-- running threads: ip and stack words -> library+offset"
            for t in $(ls /proc/$p/task); do
                [ "$(awk '{print $3}' /proc/$p/task/$t/stat)" = R ] || continue
                gdb -p "$p" -batch -ex "thread find LWP $t" >/dev/null 2>&1
                T=$(gdb -p "$p" -batch -ex "info threads" 2>/dev/null |
                    grep -E "^[ *]+[0-9]+ +Thread .*LWP $t\)" | sed -E 's/^[ *]+([0-9]+).*/\1/')
                [ -n "$T" ] || continue
                gdb -p "$p" -batch -ex "thread $T" -ex 'p/x $pc' -ex 'x/48gx $sp' 2>/dev/null |
                    grep -oE "0x[0-9a-f]{6,16}" | sort -u | while read -r a; do
                        awk -v a=$((a)) -v s="$a" '{split($1, r, "-"); b = strtonum("0x" r[1]);
                            e = strtonum("0x" r[2]); if (a >= b && a < e && $6 != "")
                            printf "   %s  %s+0x%x\n", s, $6, a - b + strtonum("0x" $3)}' /proc/$p/maps
                    done | sort -u | head -n 40
            done
            echo "-- fds:"
            for fd in $(ls /proc/$p/fd); do
                l=$(readlink /proc/$p/fd/$fd)
                case "$l" in
                anon_inode:\[eventfd\]) echo "   fd $fd eventfd $(grep -h eventfd-count /proc/$p/fdinfo/$fd)" ;;
                socket:*) ino=${l#socket:[}; ino=${ino%]}
                    echo "   fd $fd $l $(ss -tanpxe 2>/dev/null | grep -m1 "ino:$ino\b" | tr -s ' ' | cut -c1-110)" ;;
                *) echo "   fd $fd $l" ;;
                esac
            done
        } >>diag.txt 2>&1
    done
fi
"$MANA/bin/dmtcp_command" $(coord) --kill >/dev/null 2>&1
"$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
kill $pid 2>/dev/null
wait $pid 2>/dev/null
echo "diagnostics: $OUT/diag.txt"
