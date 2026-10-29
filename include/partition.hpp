#pragma once

// Contiguous row-block partitioning for distributed CSR.
//
// Each MPI rank owns a consecutive range of global rows.
// The first (N % num_ranks) ranks receive one extra row.
//
// This interface is MPI-independent.

struct RowPartition {
    int global_row_start = 0;
    int local_num_rows = 0;
};

// Compute the row ownership for a given rank.
//
// Parameters:
//   global_rows : total number of global matrix rows
//   rank        : zero-based MPI rank
//   num_ranks   : total number of MPI ranks
//
// Requirements:
//   global_rows >= 0
//   num_ranks > 0
//   0 <= rank < num_ranks
//
// Supports ranks with zero owned rows when num_ranks > global_rows.

RowPartition partition_rows(
    int global_rows,
    int rank,
    int num_ranks
);
