# ─────────────────────────────────────────────────────────────────────
#  NAS IS Benchmark — MPI (original) and NCCL (ported) builds
# ─────────────────────────────────────────────────────────────────────
#
# Usage:
#   make is_nccl          Build the NCCL version
#   make is_mpi           Build the original MPI version
#   make all              Build both
#   make run_nccl         Run the NCCL version under srun (2 ranks)
#   make run_mpi          Run the MPI version under srun (2 ranks)
#   make clean
#
#   Override CLASS on the command line:
#     make is_nccl CLASS=A
#     make is_nccl CLASS=W NPROCS=4
#
# ─────────────────────────────────────────────────────────────────────

# ── Compilers ────────────────────────────────────────────────────────
# On Perlmutter: `module load cudatoolkit` gives you nvcc;
# `cc` wraps the Cray compiler with MPI support.
NVCC   ?= nvcc
MPICC  ?= cc

# ── Problem class (S, W, A, B, C, D) ────────────────────────────────
CLASS  ?= S
NPROCS ?= 2

# ── Paths ────────────────────────────────────────────────────────────
NCCL_HOME     ?= $(NCCL_DIR)
CUDA_HOME     ?= $(CUDA_HOME)

# DMTCP (for checkpoint tests)
DMTCP_ROOT    ?= ../../../dmtcp
NCCL_PLUGIN    = ../../../bin/libncclplugin.so
CUDA_PLUGIN    = $(DMTCP_ROOT)/plugin/cuda/libdmtcp_cuda.so
DMTCP_LAUNCH   = $(DMTCP_ROOT)/bin/dmtcp_launch
DMTCP_CMD      = $(DMTCP_ROOT)/bin/dmtcp_command
DMTCP_RESTART  = $(DMTCP_ROOT)/bin/dmtcp_restart
PORT           = 7783

# ── Flags ────────────────────────────────────────────────────────────
OPTFLAGS  = -O3 -g

# NCCL build: compile .c through MPICC (which natively has MPI paths)
NCCL_CFLAGS  = $(OPTFLAGS) -DCLASS=\'$(CLASS)\' -Xcompiler -Wall
NCCL_LDFLAGS = -lnccl -lcudart

# Add include/lib paths only if set
ifneq ($(NCCL_HOME),)
  NCCL_CFLAGS  += -I$(NCCL_HOME)/include
  NCCL_LDFLAGS += -L$(NCCL_HOME)/lib
endif

# MPI build
MPI_CFLAGS   = $(OPTFLAGS) -DCLASS=\'$(CLASS)\' -Wall
MPI_LDFLAGS  =

# ── Sources ──────────────────────────────────────────────────────────
NCCL_SRCS = is_nccl.cu c_timers.c c_print_results.c
MPI_SRCS  = is.c c_timers.c c_print_results.c

NCCL_EXE  = is_nccl_exe
MPI_EXE   = is_mpi_exe

# ── Targets ──────────────────────────────────────────────────────────
.PHONY: all clean run_nccl run_mpi run_nccl_ckpt is_nccl is_mpi

all: $(NCCL_EXE) $(MPI_EXE)

is_nccl: $(NCCL_EXE)
is_mpi: $(MPI_EXE)

# --- NCCL version ---------------------------------------------------
# We link cudart/nccl manually; no MPI needed now
$(NCCL_EXE): $(NCCL_SRCS) npbparams.h c_timers.h
	$(NVCC) $(NCCL_CFLAGS) -o $@ $(NCCL_SRCS) $(NCCL_LDFLAGS)

# --- MPI version (reference) ----------------------------------------
$(MPI_EXE): $(MPI_SRCS) npbparams.h c_timers.h
	$(MPICC) $(MPI_CFLAGS) -o $@ $(MPI_SRCS) $(MPI_LDFLAGS)

# ── Run targets ──────────────────────────────────────────────────────
run_nccl: $(NCCL_EXE)
	srun -n $(NPROCS) --ntasks-per-node=$(NPROCS) --gpus-per-task=1 ./$(NCCL_EXE)

run_mpi: $(MPI_EXE)
	srun -n $(NPROCS) ./$(MPI_EXE)

# ── DMTCP checkpoint run ─────────────────────────────────────────────
run_nccl_ckpt: $(NCCL_EXE)
	-$(DMTCP_CMD) --coord-port $(PORT) --quit --quiet 2>/dev/null || true
	$(DMTCP_LAUNCH) --coord-port $(PORT) --interval 0 \
	  --with-plugin $(NCCL_PLUGIN):$(CUDA_PLUGIN) \
	  srun -n $(NPROCS) --ntasks-per-node=$(NPROCS) --gpus-per-task=1 ./$(NCCL_EXE)

# ── Cleanup ──────────────────────────────────────────────────────────
clean:
	rm -f $(NCCL_EXE) $(MPI_EXE)
	rm -f *.o *.btr
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh core*
