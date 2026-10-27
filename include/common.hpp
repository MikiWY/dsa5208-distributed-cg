#pragma once

#include <vector>

// Compressed Sparse Row (CSR) matrix.
//
// This structure stores only locally owned matrix rows.
// All column indices use global zero-based indexing.
//
// The grid dimension n is stored separately from CSRMatrix.
//
// Ownership:
//   global_rows       = total number of matrix rows (N = n*n)
//   global_row_start  = first globally owned matrix row
//   local_num_rows    = number of locally owned matrix rows
//
// Local row r corresponds to global row:
//   global_row_start + r
//
// CSR invariants:
//   1. row_ptr.size() == local_num_rows + 1
//   2. row_ptr[0] == 0
//   3. row_ptr.back() == col_idx.size() == values.size()
//   4. row_ptr is monotonically non-decreasing
//   5. 0 <= col_idx[k] < global_rows
//
// Across MPI ranks, owned row ranges must form a
// non-overlapping partition covering all global rows.
//
// Communication:
//   Allgatherv accesses global vector entries using col_idx.
//   Halo Exchange uses a separate global-to-local/ghost mapping.
//   The CSR column indices remain unchanged.

struct CSRMatrix {
    int global_rows = 0;
    int global_row_start = 0;
    int local_num_rows = 0;

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};
