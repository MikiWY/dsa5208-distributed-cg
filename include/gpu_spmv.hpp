#pragma once

#include <vector>

#include "common.hpp"

// CUDA CSR sparse matrix-vector multiplication.
//
// A must contain the complete global matrix.
// Column indices remain global and zero-based.
//
// x is the full input vector.
//
// The result is copied back to host memory.
//
// This initial correctness implementation includes
// device allocation and host-device transfers.
// Performance timing will be separated later.

std::vector<double> gpu_csr_spmv(
    const CSRMatrix& A,
    const std::vector<double>& x
);
