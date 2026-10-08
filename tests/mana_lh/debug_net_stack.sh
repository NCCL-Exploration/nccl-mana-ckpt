#!/bin/bash
# Diagnostics for two NCCL-in-the-lower-half MANA problems (2026-10-08):
#  (1) after teardown, NCCL cannot initialize any network in the lower half
#      ("Failed to initialize any NET plugin" / ncclInvalidUsage at rebuild);
#  (2) upper-half thread stacks sit outside MANA's arena (0x7fff...) and are
#      left out of the checkpoint image, so restart's clone() faults.
#
#   salloc -A m4128 -q interactive -t 00:20:00 -N 1 -C gpu --gpus-per-node=4 \
#       srun -N 1 -n 1 --overlap bash tests/mana_lh/debug_net_stack.sh

set -uo pipefail
: "${MANA:=$SCRATCH/mana-nccl-sp}"
: "${BENCH:=$SCRATCH/nccl-ckpt-report/bin/nccl_bench_mpi}"
OUT=$MANA-runs/netstack-${SLURM_JOB_ID:?run inside a Slurm allocation}
RC=$HOME/.mana-slurm-$SLURM_JOB_ID.rc
module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 nccl/2.29.2-cu13 >/dev/null 2>&1
export DARSHAN_DISABLE=1 NCCL_RAS_ENABLE=0
mkdir -p "$OUT" && cd "$OUT" || exit 1
coord() { awk '$1 == "Host:" {h = $2} $1 == "Port:" {p = $2} END {print "-h " h " -p " p}' "$RC"; }
SRUN=(srun -N 1 -n 2 --ntasks-per-node=2 --gpus-per-node=2 -c 64 --cpu-bind=cores)
ARGS=(-b allreduce -m 1048576:1048576 -T 20 -p 200)

echo "== (1a) native, NCCL_NET_PLUGIN=none NCCL_NET=Socket"
NCCL_NET_PLUGIN=none NCCL_NET=Socket NCCL_DEBUG=WARN timeout 120 "${SRUN[@]}" "$BENCH" "${ARGS[@]}" \
    -f "$OUT/id1.bin" >native_socket.log 2>&1
echo "   rc=$? $(grep -c 'validation PASSED' native_socket.log) passed"

echo "== (1b)+(2) MANA, same env, NCCL_DEBUG_SUBSYS=ALL"
"$MANA/bin/mana_coordinator" -p 0 >coord.log 2>&1
NCCL_NET_PLUGIN=none NCCL_NET=Socket NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL \
    "${SRUN[@]}" "$MANA/bin/mana_launch" --cuda "$BENCH" "${ARGS[@]}" -f "$OUT/id2.bin" \
    >mana_socket.log 2>&1 &
pid=$!
# Inspect a live rank once the application is loaded (its exe mapping exists).
for _ in $(seq 1 120); do
    p=$(pgrep -u "$USER" -f "bin/lower-half.*nccl_bench_mpi" | head -1)
    [ -n "$p" ] && grep -q nccl_bench_mpi "/proc/$p/maps" 2>/dev/null && break
    p=
    sleep 0.5
done
sleep 4
if [ -n "$p" ]; then
    {
        echo "-- live rank $p: libc instances (r-xp, offset 0)"
        awk '$6 ~ /libc\.so\.6$/ && $3 == "00000000" {print "   " $1}' /proc/$p/maps
        for r in $(awk '$6 ~ /libc\.so\.6$/ && $3 == "00000000" {print $1}' /proc/$p/maps); do
            b=0x${r%%-*}
            for f in mmap:0x1293be munmap:0x129e5e clone:0x12d69e; do
                a=$(printf "0x%x" $((b + ${f#*:})))
                echo "   [$b] ${f%%:*} @ $a: $(gdb -p "$p" -batch -ex "x/1i $a" 2>/dev/null | grep -oE ':\s+.*' | head -1)"
            done
        done
        echo "-- thread stacks (sp -> region)"
        gdb -p "$p" -batch -ex 'thread apply all p/x $sp' 2>/dev/null | grep -E '^Thread|^\$' | paste - - |
            sed -E 's/.*LWP ([0-9]+)\) "([^"]*)".*= (0x[0-9a-f]+)/\1 \2 \3/' | while read -r t n sp; do
            echo "   $t $n $sp -> $(awk -v a=$((sp)) '{split($1,r,"-"); b=strtonum("0x"r[1]); e=strtonum("0x"r[2]); if (a>=b && a<e) print $1, $2, $6}' /proc/$p/maps)"
        done
        echo "-- arena hint: first anonymous rw region above the main stack"
        grep -m3 -E "rw-p 00000000 00:00 0" /proc/$p/maps | sed 's/^/   /'
    } >live.txt 2>&1
fi
wait $pid
echo "   MANA rc=$? $(grep -c 'validation PASSED' mana_socket.log) passed"
"$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
echo "-- NET lines of the MANA run:" >net.txt
grep -aE "NET/|net\.cc|Socket|socket|WARN|ifaddr|interface|IfName|Bootstrap" mana_socket.log | head -40 >>net.txt
echo "results: $OUT (live.txt, net.txt)"
