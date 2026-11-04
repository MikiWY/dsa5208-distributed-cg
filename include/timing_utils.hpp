#pragma once

#include <mpi.h>

#include "solver_timing.hpp"

// Lightweight timing helpers for MPI CG.
//
// Timing is optional: nullptr disables component accumulation.
//
// These helpers do not introduce MPI synchronization.

inline double timing_now(const SolverTiming* timing) {
    return timing ? MPI_Wtime() : 0.0;
}

inline void timing_add_compute(
    SolverTiming* timing,
    double start
) {
    if (timing) {
        timing->compute_s += MPI_Wtime() - start;
    }
}

inline void timing_add_vector_comm(
    SolverTiming* timing,
    double start
) {
    if (timing) {
        timing->vector_comm_s += MPI_Wtime() - start;
    }
}

inline void timing_add_reduction(
    SolverTiming* timing,
    double start
) {
    if (timing) {
        timing->reduction_s += MPI_Wtime() - start;
    }
}
