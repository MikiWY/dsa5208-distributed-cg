#include "partition.hpp"

#include <algorithm>
#include <stdexcept>

RowPartition partition_rows(
    int global_rows,
    int rank,
    int num_ranks
) {
    if (global_rows < 0 ||
        num_ranks <= 0 ||
        rank < 0 ||
        rank >= num_ranks) {
        throw std::invalid_argument(
            "Invalid row partition parameters"
        );
    }

    const int base = global_rows / num_ranks;
    const int remainder = global_rows % num_ranks;

    RowPartition partition;

    partition.local_num_rows =
        base + (rank < remainder ? 1 : 0);

    partition.global_row_start =
        rank * base + std::min(rank, remainder);

    return partition;
}
