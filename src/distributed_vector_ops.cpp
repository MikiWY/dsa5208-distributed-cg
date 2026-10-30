#include "distributed_vector_ops.hpp"

#include <stdexcept>

double distributed_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm
) {
    // Validate local vector dimensions.
    const int local_valid =
        (a_local.size() == b_local.size()) ? 1 : 0;

    // All ranks must agree that the inputs are valid
    // before proceeding to the next collective operation.
    int global_valid = 0;

    MPI_Allreduce(
        &local_valid,
        &global_valid,
        1,
        MPI_INT,
        MPI_MIN,
        comm
    );

    if (global_valid == 0) {
        throw std::invalid_argument(
            "Distributed dot product vector sizes do not match"
        );
    }

    // Compute the local dot product.
    double local_sum = 0.0;

    for (std::size_t i = 0; i < a_local.size(); ++i) {
        local_sum += a_local[i] * b_local[i];
    }

    // Sum local contributions across all MPI ranks.
    double global_sum = 0.0;

    MPI_Allreduce(
        &local_sum,
        &global_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        comm
    );

    return global_sum;
}
