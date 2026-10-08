#!/bin/bash
# Where "libmana.so: undefined symbol: , version GLIBC_2.2.5" comes from
# (~/mana-nccl-sp b94003d on Perlmutter, 2026-10-08).
#  (1) lazy binding: gdb on rank 0 stops at the upper half's ld.so
#      _dl_fatal_printf (or exit_group) during the checkpoint, and dumps the
#      failing thread: registers, stack words -> library+offset, mappings.
#      The PLT caller's return address and the pushed relocation index sit
#      just above _dl_runtime_resolve's save area.
#  (2) LD_BIND_NOW=1 with LD_DEBUG=bindings: the last bindings ld.so made
#      before the error, and whether the program got past loading.
#
#   salloc ... srun -N 1 -n 1 --overlap bash tests/mana_lh/debug_fatal.sh

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${MANA:=$SCRATCH/mana-nccl-sp}"
: "${BENCH:=$SCRATCH/nccl-ckpt-report/bin/nccl_bench_mpi}"
: "${STEPS:=1 2}"
OUT=$MANA-runs/fatal-${SLURM_JOB_ID:?run inside a Slurm allocation}
RC=$HOME/.mana-slurm-$SLURM_JOB_ID.rc
LDSO=/lib64/ld-linux-x86-64.so.2
module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 nccl/2.29.2-cu13 >/dev/null 2>&1
export DARSHAN_DISABLE=1 NCCL_RAS_ENABLE=0 NCCL_DEBUG=WARN
mkdir -p "$OUT" && cd "$OUT" || exit 1
coord() { awk '$1 == "Host:" {h = $2} $1 == "Port:" {p = $2} END {print "-h " h " -p " p}' "$RC"; }
SRUN=(srun -N 1 -n 2 --ntasks-per-node=2 --gpus-per-node=2 -c 64 --cpu-bind=cores)
ARGS=(-b allreduce -m 4194304:4194304 -T 60 -p 200)
sym_off() { readelf -sW "$LDSO" | awk -v s="$1" '$8 == s || $8 == s "@@GLIBC_PRIVATE" {print "0x" $2; exit}'; }

if [[ " $STEPS " == *" 1 "* ]]; then
    echo "== (1) lazy binding, gdb on rank 0 through the checkpoint"
    rm -rf ckpt_rank_* dmtcp_restart_script*.sh
    "$MANA/bin/mana_coordinator" -p 0 >coord1.log 2>&1
    "${SRUN[@]}" "$MANA/bin/mana_launch" --cuda "$BENCH" "${ARGS[@]}" -f "$OUT/id1.bin" >launch1.log 2>&1 &
    pid=$!
    for _ in $(seq 1 300); do grep -q "^# Size" launch1.log 2>/dev/null && break; sleep 0.5; done
    sleep 3
    p=$(pgrep -u "$USER" -f "bin/lower-half.*nccl_bench_mpi" | sort -n | head -1)
    cp "/proc/$p/maps" maps1.txt
    # Break in every ld.so instance (the lower half's and the upper half's).
    fp=$(sym_off _dl_fatal_printf); se=$(sym_off _dl_signal_error)
    {
        echo "set pagination off"
        echo "set confirm off"
        echo "handle all nostop noprint pass"
        awk -v f="$LDSO" '$6 == f && $3 == "00000000" {print "0x" substr($1, 1, index($1, "-") - 1)}' maps1.txt |
            while read -r b; do
                printf 'break *0x%x\n' $((b + fp))
            done
        echo "catch syscall exit_group"
        echo "continue"
        echo "echo \n######## STOPPED\n"
        echo "info threads"
        echo "info registers rip rsp rbp rdi rsi rdx fs_base"
        echo "x/1536gx \$sp"
        echo "info proc mappings"
        echo "detach"
        echo "quit"
    } >gdb1.cmd
    echo "   rank pid $p; _dl_fatal_printf +$fp _dl_signal_error +$se; breakpoints: $(grep -c '^break' gdb1.cmd)"
    timeout 240 gdb -p "$p" -batch -x gdb1.cmd >gdb1.out 2>&1 &
    gpid=$!
    sleep 5
    rm -f dmtcp_restart_script*.sh
    "$MANA/bin/dmtcp_command" $(coord) -c
    wait $gpid
    echo "   gdb rc=$? ($(grep -c . gdb1.out) lines); stop: $(grep -m1 -E '^(Thread|Catchpoint|Breakpoint) .*(hit|exit_group)|^Catchpoint|^Thread .* hit' gdb1.out)"
    grep -m2 -E "symbol lookup error|Segmentation|JASSERT" launch1.log | sed 's/^/   /'
    "$MANA/bin/dmtcp_command" $(coord) --kill >/dev/null 2>&1
    "$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    # Stack words that point into a file mapping, as library+offset.
    awk 'FNR == NR {split($1, r, "-"); n++; b[n] = strtonum("0x" r[1]); e[n] = strtonum("0x" r[2]);
                    o[n] = strtonum("0x" $3); f[n] = $6; next}
         /^0x[0-9a-f]+:/ {for (i = 2; i <= NF; i++) {v = strtonum($i); if (v < 4096) continue;
                    for (k = 1; k <= n; k++) if (v >= b[k] && v < e[k] && f[k] != "") {
                        printf "%s[+%d] %s  %s+0x%x\n", $1, i - 2, $i, f[k], v - b[k] + o[k]; break}}}' \
        maps1.txt gdb1.out >stack1.txt
    echo "   stack words in file mappings: $(wc -l <stack1.txt) (stack1.txt)"
fi

if [[ " $STEPS " == *" 2 "* ]]; then
    echo "== (2) LD_BIND_NOW=1 LD_DEBUG=bindings"
    mkdir -p lddbg && rm -f lddbg/*.*
    "$MANA/bin/mana_coordinator" -p 0 >coord2.log 2>&1
    timeout 180 "${SRUN[@]}" --export=ALL,LD_BIND_NOW=1,LD_DEBUG=bindings,LD_DEBUG_OUTPUT=$OUT/lddbg/b \
        "$MANA/bin/mana_launch" --cuda "$BENCH" "${ARGS[@]}" -T 10 -f "$OUT/id2.bin" >launch2.log 2>&1
    echo "   rc=$? $(grep -c 'validation PASSED' launch2.log) passed; $(grep -c '^# Size' launch2.log) headers"
    grep -m2 "symbol lookup error" launch2.log | sed 's/^/   /'
    "$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
    # The rank's file is the one that mentions libmana.so and the benchmark.
    for f in lddbg/b.*; do
        grep -q "nccl_bench_mpi" "$f" && grep -q "libmana.so" "$f" || continue
        echo "-- $f: $(wc -l <"$f") lines; last 15 bindings:"
        grep "binding file" "$f" | tail -15 | sed 's/^/   /'
        echo "   bindings from libmana.so: $(grep -c 'binding file [^ ]*libmana.so' "$f")"
    done >lddbg.txt
    head -60 lddbg.txt
fi
if [[ " $STEPS " == *" 3 "* ]]; then
    echo "== (3) libmana.so in memory vs on disk (live rank, no checkpoint)"
    "$MANA/bin/mana_coordinator" -p 0 >coord3.log 2>&1
    "${SRUN[@]}" "$MANA/bin/mana_launch" --cuda "$BENCH" "${ARGS[@]}" -T 20 -f "$OUT/id3.bin" >launch3.log 2>&1 &
    pid=$!
    for _ in $(seq 1 300); do grep -q "^# Size" launch3.log 2>/dev/null && break; sleep 0.5; done
    p=$(pgrep -u "$USER" -f "bin/lower-half.*nccl_bench_mpi" | sort -n | head -1)
    PY=$(command -v python3.11 || command -v python3)
    "$PY" "$HERE/memdiff.py" "$p" lib/dmtcp/libmana.so >memdiff3.txt 2>&1
    "$PY" "$HERE/memdiff.py" "$p" libc.so.6 >memdiff3_libc.txt 2>&1
    cat memdiff3.txt | head -40
    echo "-- libc.so.6 (expected: the mmap/munmap/_exit/__sbrk patches):"
    head -30 memdiff3_libc.txt
    wait $pid
    "$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
fi
echo "results: $OUT"
