# nccl-mana-ckpt agent map

## paths

| area | files |
|------|-------|
| plugin hooks | `src/libncclplugin.c` |
| NCCL wrappers | `src/nccl_wrappers.c`, `include/nccl_wrappers.h` |
| in-flight state | `src/inflight.c`, `src/inflight.h` |
| KVDB bridge | `src/dmtcp_kvdb_c.cpp`, `src/dmtcp_kvdb_c.h` |
| root build/checks | `Makefile` |
| main tests | `tests/Makefile`, `tests/nccl_test_driver.cpp`, `tests/single_gpu_test.c` |
| extra single-GPU | `single_gpu_tests/Makefile`, `single_gpu_tests/` |
| extra multi-GPU | `multiple_gpu_tests/Makefile`, `multiple_gpu_tests/` |
| docs | `README.md`, `AGENTS.md` |
| deps | `dmtcp/`, `crac/` |

## entrypoints

| command | purpose |
|---------|---------|
| `make` / `make plugin` | build `src/libncclplugin.so` |
| `make nccl_check` | root single-GPU DMTCP launch, NCCL plugin only |
| `make drain_recv_check` / `make drain_send_check` | root MPI drain checks |
| `make cc_simple_check` / `make cc_complex_check` | root collective checks |
| `cd tests && make test_all` | NCCL+CRAC single-GPU criteria |
| `cd tests && make test1`..`test4` | individual runnable criteria |
| `cd multiple_gpu_tests && make test_all` | multi-GPU criteria |
| `cd multiple_gpu_tests && make gather_p2p_test` | `ncclGather_p2p` correctness |
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
| single-GPU tests | `tests/nccl_test_driver.cpp`, `tests/Makefile` |
| multi-GPU/P2P tests | `multiple_gpu_tests/nccl_p2p_tests.cpp`, `multiple_gpu_tests/Makefile` |
| build/env changes | root `Makefile`, then test Makefiles |
| user docs | `README.md` |

## gotchas

- NCCL `-L` paths are hardcoded in Makefiles.
- Defaults: root `Makefile` uses `${CURDIR}/dmtcp` and `${CURDIR}/crac`; test Makefiles use `../dmtcp` and `../crac`.
- Full CUDA checkpoint/restart flows need both `src/libncclplugin.so` and `crac/libdmtcp_crac.so`.
- Some root check targets currently load only the NCCL plugin.
- `tests/Makefile` declares `test5` phony but has no `test5` recipe; document/run `test1`..`test4`.
- Preserve user changes in dirty worktrees; do not revert unrelated `multiple_gpu_tests/` edits.
