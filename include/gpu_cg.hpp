#pragma once

#include <vector>

#include "cg.hpp"
#include "common.hpp"

// Single-GPU CUDA Conjugate Gradient solver.
//
// Requirements:
//   A contains the complete global CSR matrix.
//   A is symmetric positive definite.
//   b contains the complete RHS vector.
//   Initial guess x0 = 0.
//
// GPU execution:
//   CSR matrix and CG vectors remain on the device.
//   SpMV, dot products and vector updates execute on GPU.
//   Only scalar convergence information is transferred
//   to the host during iterations.
//
// Returns the shared SolverResult structure.
//
// This initial version targets correctness.
// GPU benchmark timing is implemented separately.

SolverResult gpu_conjugate_gradient(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const SolverOptions& options
);
