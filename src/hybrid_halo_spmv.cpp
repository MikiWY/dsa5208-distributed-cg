#include "hybrid_halo_spmv.hpp"

#include <omp.h>

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Hybrid MPI + OpenMP SpMV using blocking Halo Exchange.
std::vector<double> hybrid_spmv_halo(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const HaloExchangePlan& plan,
    MPI_Comm comm
) {
    // MPI calls must be made by the thread that initialized MPI.
    int provided = MPI_THREAD_SINGLE;
    int is_main_thread = 0;

    MPI_Query_thread(&provided);
    MPI_Is_thread_main(&is_main_thread);

    const bool thread_valid =
        provided >= MPI_THREAD_FUNNELED &&
        is_main_thread != 0 &&
        !omp_in_parallel();

    const auto& vector_positions = plan.vector_positions;

    bool valid =
        thread_valid &&
        A.global_rows == plan.global_rows &&
        A.global_row_start == plan.global_row_start &&
        A.local_num_rows == plan.local_num_rows &&
        A.local_num_rows >= 0 &&
        A.row_ptr.size() ==
            static_cast<std::size_t>(A.local_num_rows) + 1 &&
        A.col_idx.size() == A.values.size() &&
        vector_positions.size() == A.col_idx.size() &&
        x_local.size() ==
            static_cast<std::size_t>(A.local_num_rows);

    if (valid) {
        valid =
            !A.row_ptr.empty() &&
            A.row_ptr.front() == 0 &&
            A.row_ptr.back() >= 0 &&
            static_cast<std::size_t>(A.row_ptr.back()) ==
                A.col_idx.size();
    }

    if (valid) {
        for (int r = 0; r < A.local_num_rows; ++r) {
            if (A.row_ptr[r] < 0 ||
                A.row_ptr[r] > A.row_ptr[r + 1] ||
                A.row_ptr[r + 1] > A.row_ptr.back()) {
                valid = false;
                break;
            }
        }
    }

    if (valid) {
        for (int position : vector_positions) {
            const long long ghost =
                -static_cast<long long>(position) - 1;

            if (position >= 0) {
                if (static_cast<std::size_t>(position) >=
                    x_local.size()) {
                    valid = false;
                    break;
                }
            } else if (ghost < 0 ||
                       static_cast<std::size_t>(ghost) >=
                           plan.recv_global_indices.size()) {
                valid = false;
                break;
            }
        }
    }

    const int local_ok = valid ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, comm
    );

    if (!global_ok) {
        throw std::invalid_argument(
            "Invalid Hybrid Halo SpMV inputs"
        );
    }

    // MPI communication remains outside OpenMP regions.
    const std::vector<double> ghost_values =
        exchange_halo_values_blocking(
            plan, x_local, comm
        );

    std::vector<double> y_local(A.local_num_rows, 0.0);

    // Phase 2: thread-safe local CSR multiplication.
    // Each thread writes independent output rows.
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < A.local_num_rows; ++r) {
        double sum = 0.0;

        for (int k = A.row_ptr[r];
             k < A.row_ptr[r + 1]; ++k) {
            const int position = vector_positions[k];

            const double x_value =
                position >= 0
                    ? x_local[position]
                    : ghost_values[-position - 1];

            sum += A.values[k] * x_value;
        }

        y_local[r] = sum;
    }

    return y_local;
}
