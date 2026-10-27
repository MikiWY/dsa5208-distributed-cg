#pragma once

#include "common.hpp"

// Generate locally owned CSR rows of the 2D Poisson matrix.
//
// Mathematical specification:
//   N = n * n
//   h = 1 / (n + 1)
//   diagonal = 4 / (h * h)
//   valid neighbor = -1 / (h * h)
//
// Global indexing:
//   k = j * n + i
//
// Parameters:
//   n                Number of interior grid points per dimension
//   global_row_start First globally owned matrix row
//   local_num_rows   Number of locally owned matrix rows
//
// Returns:
//   CSRMatrix with local row pointers and global column indices.
//
// Connections across MPI rank boundaries are preserved.
// Connections outside the physical grid are excluded.
//
// This function does not perform MPI communication.

CSRMatrix generate_poisson_matrix(
    int n,
    int global_row_start,
    int local_num_rows
);
