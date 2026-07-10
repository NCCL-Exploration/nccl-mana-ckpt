# nccl-mana-ckpt agent map

## paths

| area | files |
|------|-------|
| plugin hooks | `src/libncclplugin.c` |
| NCCL wrappers | `src/nccl_wrappers.c`, `include/nccl_wrappers.h` |
| in-flight state | `src/inflight.c`, `src/inflight.h` |
| KVDB bridge | `src/dmtcp_kvdb_c.cpp`, `src/dmtcp_kvdb_c.h` |
| root build/checks | `Makefile` |
| delegator tests | `tests/Makefile` |
| single-GPU tests | `tests/single/Makefile`, `tests/single/nccl_test_driver.cpp`, `tests/single/single_gpu_test.c` |
| multi-GPU tests | `tests/multi/Makefile`, `tests/multi/nccl_p2p_tests.cpp`, `tests/multi/nccl_test_driver_multiple.cpp` |
| docs | `README.md`, `AGENTS.md` |
| deps | `dmtcp/` |

## entrypoints

| command | purpose |
|---------|---------|
| `make` / `make plugin` | build `bin/libncclplugin.so` |
| `make nccl_check` | root single-GPU DMTCP launch, NCCL plugin only |
| `make drain_recv_check` / `make drain_send_check` | root MPI drain checks |
| `make cc_simple_check` / `make cc_complex_check` | root collective checks |
| `cd tests && make test_all` | NCCL+CUDA single-GPU criteria |
| `cd tests && make test1`..`test4` | individual runnable criteria |
| `cd tests/multi && make test_all` | multi-GPU criteria |
| `cd tests/multi && make gather_p2p_test` | `ncclGather_p2p` correctness |
| `cd tests/multi && make scatter_p2p_test` | `ncclScatter_p2p` correctness |
| `make clean` | clean artifacts for current Makefile |

## routing

| task | start |
|------|-------|
| DMTCP event behavior | `src/libncclplugin.c` |
| checkpoint drain/reset | `src/libncclplugin.c`, then `src/inflight.c` |
| send/recv tracking | `src/nccl_wrappers.c`, then `src/inflight.c` |
| collective sequence tracking | `src/nccl_wrappers.c`, `src/libncclplugin.c` |
| communicator/rank bugs | `src/nccl_wrappers.c`, `include/nccl_wrappers.h` |
| KVDB coordination | `src/dmtcp_kvdb_c.cpp`, callsites in `src/libncclplugin.c` |
| single-GPU tests | `tests/single/nccl_test_driver.cpp`, `tests/single/Makefile` |
| multi-GPU/P2P tests | `tests/multi/nccl_p2p_tests.cpp`, `tests/multi/Makefile` |
| build/env changes | root `Makefile`, then test Makefiles |
| user docs | `README.md` |

## gotchas

- NCCL `-L` paths are hardcoded in Makefiles.
- Defaults: root `Makefile` uses `${CURDIR}/dmtcp`; test Makefiles use `../dmtcp`.
- Full CUDA checkpoint/restart flows need both `bin/libncclplugin.so` and the CUDA plugin (`libdmtcp_cuda.so`).
- Some root check targets currently load only the NCCL plugin.
- `tests/multi/nccl_p2p_tests.cpp` compares `ncclGather_p2p` against real `ncclAllGather`; `ncclScatter_p2p` is checked against the deterministic rank-slot input pattern because this NCCL copy does not provide native `ncclScatter`.
- `tests/single/Makefile` declares `test5` phony but has no `test5` recipe; document/run `test1`..`test4`.
- Preserve user changes in dirty worktrees; do not revert unrelated `tests/multi/` edits.
