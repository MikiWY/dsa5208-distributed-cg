#include "halo_spmv.hpp"

#include <stdexcept>
#include <unordered_map>
#include <vector>

// MPI-only CSR SpMV using blocking Halo Exchange.
std::vector<double> distributed_spmv_halo(
    const CSRMatrix& A,
    const std::vector<double>& x_local,
    const HaloExchangePlan& plan,
    MPI_Comm comm
) {
    // The exchange function performs collective input checks.
    const std::vector<double> ghost_values =
        exchange_halo_values_blocking(
            plan, x_local, comm
        );

    // Build global-index-to-ghost-position mapping.
    std::unordered_map<int, std::size_t> ghost_lookup;

    ghost_lookup.reserve(plan.recv_global_indices.size());

    for (std::size_t i = 0;
         i < plan.recv_global_indices.size(); ++i) {
        ghost_lookup.emplace(
            plan.recv_global_indices[i], i
        );
    }

    std::vector<double> y_local(
        A.local_num_rows, 0.0
    );

    const int local_start = A.global_row_start;
    const int local_end =
        local_start + A.local_num_rows;

    for (int r = 0; r < A.local_num_rows; ++r) {
        double sum = 0.0;

        for (int k = A.row_ptr[r];
             k < A.row_ptr[r + 1]; ++k) {
            const int col = A.col_idx[k];

            double x_value = 0.0;

            if (col >= local_start && col < local_end) {
                x_value = x_local[col - local_start];
            } else {
                const auto it = ghost_lookup.find(col);

                if (it == ghost_lookup.end()) {
                    throw std::invalid_argument(
                        "Missing ghost value for CSR column"
                    );
                }

                x_value = ghost_values[it->second];
            }

            sum += A.values[k] * x_value;
        }

        y_local[r] = sum;
    }

    return y_local;
}
