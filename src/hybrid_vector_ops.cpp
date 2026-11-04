#include "hybrid_vector_ops.hpp"

#include <stdexcept>

#include "openmp_vector_ops.hpp"
#include "timing_utils.hpp"

double hybrid_global_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm,
    SolverTiming* timing
) {
    if (a_local.size() != b_local.size()) {
        throw std::invalid_argument(
            "Hybrid dot product vector sizes do not match"
        );
    }

    const double compute_start = timing_now(timing);

    const double local_sum =
        openmp_dot_product(a_local, b_local);

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
double hybrid_global_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm
) {
    return hybrid_global_dot_product(
        a_local, b_local, comm, nullptr
    );
}
