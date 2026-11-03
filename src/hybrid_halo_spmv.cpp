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

    // Check local metadata before any vector communication.
    const bool metadata_valid =
        A.global_rows == plan.global_rows &&
        A.global_row_start == plan.global_row_start &&
        A.local_num_rows == plan.local_num_rows &&
        A.local_num_rows >= 0 &&
        A.row_ptr.size() ==
            static_cast<std::size_t>(A.local_num_rows) + 1 &&
        A.col_idx.size() == A.values.size();

    const int local_ok =
        (thread_valid && metadata_valid) ? 1 : 0;

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

    // Phase 1: MPI communication outside OpenMP parallel regions.
    const std::vector<double> ghost_values =
        exchange_halo_values_blocking(
            plan, x_local, comm
        );

    // Map global ghost indices to received buffer positions.
    std::unordered_map<int, std::size_t> ghost_lookup;

    ghost_lookup.reserve(plan.recv_global_indices.size());

    for (std::size_t i = 0;
         i < plan.recv_global_indices.size(); ++i) {
        ghost_lookup.emplace(
            plan.recv_global_indices[i], i
        );
    }

    const int local_start = A.global_row_start;
    const int local_end =
        local_start + A.local_num_rows;

    // Precompute vector positions before entering OpenMP.
    // Local positions are nonnegative.
    // Ghost positions are encoded as -(index + 1).
    std::vector<int> vector_positions(A.col_idx.size());

    bool positions_valid = true;

    for (std::size_t k = 0; k < A.col_idx.size(); ++k) {
        const int col = A.col_idx[k];

        if (col >= local_start && col < local_end) {
            vector_positions[k] = col - local_start;
        } else {
            const auto it = ghost_lookup.find(col);

            if (it == ghost_lookup.end()) {
                positions_valid = false;
                break;
            }

            vector_positions[k] =
                -static_cast<int>(it->second) - 1;
        }
    }

    // Coordinate any missing ghost mapping before parallel work.
    const int position_ok = positions_valid ? 1 : 0;
    int global_position_ok = 0;

    MPI_Allreduce(
        &position_ok,
        &global_position_ok,
        1, MPI_INT, MPI_MIN, comm
    );

    if (!global_position_ok) {
        throw std::invalid_argument(
            "Invalid Hybrid Halo ghost mapping"
        );
    }

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
