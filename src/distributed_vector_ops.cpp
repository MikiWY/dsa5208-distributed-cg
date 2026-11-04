#include "distributed_vector_ops.hpp"

#include <stdexcept>

#include "timing_utils.hpp"

double distributed_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm,
    SolverTiming* timing
) {
    const int local_valid =
        (a_local.size() == b_local.size()) ? 1 : 0;

    int global_valid = 0;

    const double check_start = timing_now(timing);

    MPI_Allreduce(
        &local_valid, &global_valid,
        1, MPI_INT, MPI_MIN, comm
    );

    timing_add_reduction(timing, check_start);

    if (!global_valid) {
        throw std::invalid_argument(
            "Distributed dot product vector sizes do not match"
        );
    }

    const double compute_start = timing_now(timing);

    double local_sum = 0.0;

    for (std::size_t i = 0; i < a_local.size(); ++i) {
        local_sum += a_local[i] * b_local[i];
    }

    timing_add_compute(timing, compute_start);

    double global_sum = 0.0;

    const double reduction_start = timing_now(timing);

    MPI_Allreduce(
        &local_sum, &global_sum,
        1, MPI_DOUBLE, MPI_SUM, comm
    );

    timing_add_reduction(timing, reduction_start);

    return global_sum;
}

// Preserve the original API.
double distributed_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm
) {
    return distributed_dot_product(
        a_local, b_local, comm, nullptr
    );
}
