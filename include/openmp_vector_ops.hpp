#pragma once

#include <vector>

#include "common.hpp"

// Compute local CSR SpMV using OpenMP.
//
// A contains locally owned CSR rows.
// x_global contains the complete global input vector.
// CSR column indices are global.
//
// Each OpenMP thread computes independent output rows.
// No MPI calls are made inside this function.

std::vector<double> openmp_csr_spmv(
    const CSRMatrix& A,
    const std::vector<double>& x_global
);

// Compute the local dot product using OpenMP.
//
// The two vectors must have equal lengths.
// Uses an OpenMP reduction to avoid data races.
//
// This function performs no MPI communication.
// The caller is responsible for MPI_Allreduce
// when a global dot product is required.

double openmp_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local
);

// Perform y = y + alpha * x using OpenMP.
//
// x is read-only; y is updated in place.
// Both vectors must have equal lengths.
//
// Each thread updates independent elements of y.
// This function performs no MPI communication.
//
// x and y should be distinct vectors for the intended
// use in CG. Overlapping storage is not supported.

void openmp_axpy(
    double alpha,
    const std::vector<double>& x,
    std::vector<double>& y
);
