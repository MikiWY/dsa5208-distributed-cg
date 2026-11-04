#pragma once

// Accumulated CG-only timing components.
//
// All values are measured in seconds.
//
// compute_s:
//   Local sparse matrix-vector multiplication,
//   local dot products, and vector updates.
//
// vector_comm_s:
//   MPI_Allgatherv or Halo value communication.
//
// reduction_s:
//   MPI global reductions used by CG.
//
// Timings are accumulated per MPI rank.
// The benchmark later reports the maximum per-rank
// accumulated time for each category.

struct SolverTiming {
    double compute_s = 0.0;
    double vector_comm_s = 0.0;
    double reduction_s = 0.0;
};
