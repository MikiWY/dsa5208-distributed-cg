#include "hybrid_vector_ops.hpp"

#include <stdexcept>

#include "openmp_vector_ops.hpp"

// Compute the global dot product using OpenMP + MPI.
double hybrid_global_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm
) {
    // Local validation only.
    // The calling solver must collectively validate dimensions
    // before entering this function.
    if (a_local.size() != b_local.size()) {
        throw std::invalid_argument(
            "Hybrid dot product vector sizes do not match"
        );
    }

    // OpenMP local computation.
    const double local_sum =
        openmp_dot_product(a_local, b_local);

    // MPI communication is performed outside OpenMP regions
    // by the thread that initialized MPI.
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
