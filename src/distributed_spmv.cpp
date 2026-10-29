#include "distributed_spmv.hpp"

#include <stdexcept>
#include <vector>

std::vector<double> distributed_spmv_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const std::vector<int>& recvcounts,
    const std::vector<int>& displs,
    MPI_Comm comm
) {
    int rank = 0;
    int size = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Validate local input dimensions.
    if (A.global_rows < 0 ||
        A.global_row_start < 0 ||
        A.local_num_rows < 0 ||
        A.global_row_start > A.global_rows ||
        A.local_num_rows > A.global_rows - A.global_row_start ||
        x_local.size() !=
            static_cast<std::size_t>(A.local_num_rows) ||
        recvcounts.size() != static_cast<std::size_t>(size) ||
        displs.size() != static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            "Invalid distributed SpMV dimensions"
        );
    }

    // Validate collective receive layout.
    int expected_start = 0;

    for (int r = 0; r < size; ++r) {
        if (recvcounts[r] < 0 ||
            displs[r] != expected_start ||
            recvcounts[r] > A.global_rows - expected_start) {
            throw std::invalid_argument(
                "Invalid Allgatherv receive layout"
            );
        }

        expected_start += recvcounts[r];
    }

    if (expected_start != A.global_rows ||
        displs[rank] != A.global_row_start ||
        recvcounts[rank] != A.local_num_rows) {
        throw std::invalid_argument(
            "Allgatherv layout does not match local ownership"
        );
    }

    // Gather all distributed vector segments.
    std::vector<double> x_global(A.global_rows, 0.0);

    MPI_Allgatherv(
        x_local.data(),
        A.local_num_rows,
        MPI_DOUBLE,
        x_global.data(),
        recvcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        comm
    );

    // Compute locally owned output rows.
    std::vector<double> y_local(A.local_num_rows, 0.0);

    for (int r = 0; r < A.local_num_rows; ++r) {
        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            y_local[r] +=
                A.values[p] * x_global[A.col_idx[p]];
        }
    }

    return y_local;
}
