/* CLASS is set via -DCLASS='X' on the compiler command line.
   If not set, default to 'S' (small) for safety.              */
#ifndef CLASS
#define CLASS 'S'
#endif

/*
   This file is generated automatically by the setparams utility.
   It sets the number of processors and the class of the NPB
   in this directory. Do not modify it by hand.   */

#define COMPILETIME "07 Jun 2026"
#define NPBVERSION "3.4.4"
#define MPICC "nvcc+mpi"
#define CFLAGS "-O3"
#define CLINK "nvcc"
#define CLINKFLAGS "-O3 -lnccl -lmpi -lcudart"
#define CMPI_LIB "(none)"
#define CMPI_INC "(none)"
