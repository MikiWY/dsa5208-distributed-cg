#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "hybrid_vector_ops.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;

    MPI_Init_thread(
        &argc, &argv,
        MPI_THREAD_FUNNELED, &provided
    );

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 4;

    RowPartition part = partition_rows(N, rank, size);

    std::vector<double> a_local(part.local_num_rows);
    std::vector<double> b_local(part.local_num_rows, 2.0);

    for (int i = 0; i < part.local_num_rows; ++i) {
        a_local[i] =
            static_cast<double>(part.global_row_start + i + 1);
    }

    // OpenMP local dot product + MPI global reduction.
    const double result = hybrid_global_dot_product(
        a_local, b_local, MPI_COMM_WORLD
    );

    const bool local_pass =
        provided >= MPI_THREAD_FUNNELED &&
        std::isfinite(result) &&
        std::abs(result - 20.0) <= 1e-12;

    int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "MPI ranks: " << size
                  << ", OpenMP threads: "
                  << omp_get_max_threads() << "\n";

        std::cout << "Global dot product: "
                  << result << "\n";

        std::cout << (global_ok ? "PASS" : "FAIL")
                  << ": Hybrid global dot product\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
