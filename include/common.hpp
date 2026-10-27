#pragma once

#include <vector>

// Compressed Sparse Row (CSR) matrix.
//
// Stores only locally owned matrix rows.
// Column indices always refer to global unknown indices.
//
// For serial execution:
//   global_row_start = 0
//   local_num_rows = global_rows

struct CSRMatrix {
    int global_rows = 0;
    int global_row_start = 0;
    int local_num_rows = 0;

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};
