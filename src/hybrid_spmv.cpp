#include "hybrid_spmv.hpp"
#include "timing_utils.hpp"

#include <stdexcept>
#include <vector>

#include "openmp_vector_ops.hpp"

std::vector<double> hybrid_spmv_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const std::vector<int>& recvcounts,
    const std::vector<int>& displs,
    MPI_Comm comm,
    SolverTiming* timing
) {
    int rank = 0;
    int size = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Validate local dimensions.
    const bool local_valid =
        A.global_rows > 0 &&
        A.global_row_start >= 0 &&
        A.local_num_rows >= 0 &&
        A.global_row_start <= A.global_rows &&
        A.local_num_rows <= A.global_rows - A.global_row_start &&
        x_local.size() ==
            static_cast<std::size_t>(A.local_num_rows) &&
        recvcounts.size() == static_cast<std::size_t>(size) &&
        displs.size() == static_cast<std::size_t>(size);

    int local_ok = local_valid ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok, 1,
        MPI_INT, MPI_MIN, comm
    );

    if (!global_ok) {
        throw std::invalid_argument(
            "Invalid Hybrid SpMV dimensions"
        );
    }

    // Check the contiguous Allgatherv layout.
    int expected_start = 0;
    bool layout_valid = true;

    for (int r = 0; r < size; ++r) {
        if (recvcounts[r] < 0 ||
            displs[r] != expected_start ||
            recvcounts[r] > A.global_rows - expected_start) {
            layout_valid = false;
            break;
        }

        expected_start += recvcounts[r];
    }

    if (expected_start != A.global_rows ||
        displs[rank] != A.global_row_start ||
        recvcounts[rank] != A.local_num_rows) {
        layout_valid = false;
    }

    local_ok = layout_valid ? 1 : 0;

    MPI_Allreduce(
        &local_ok, &global_ok, 1,
        MPI_INT, MPI_MIN, comm
    );

    if (!global_ok) {
        throw std::invalid_argument(
            "Invalid Hybrid Allgatherv layout"
        );
    }

    // Gather the full input vector on every MPI rank.
    std::vector<double> x_global(A.global_rows, 0.0);

    const double comm_start = timing_now(timing);
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

    timing_add_vector_comm(timing, comm_start);

    const double compute_start = timing_now(timing);
    // Local CSR computation uses OpenMP.
    auto y_local = openmp_csr_spmv(A, x_global);
    timing_add_compute(timing, compute_start);
    return y_local;
}

// Compatibility overload without component timing.
std::vector<double> hybrid_spmv_allgatherv(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const std::vector<int>& recvcounts,
    const std::vector<int>& displs,
    MPI_Comm comm
) {
    return hybrid_spmv_allgatherv(
        A, x_local, recvcounts, displs, comm, nullptr
    );
}
