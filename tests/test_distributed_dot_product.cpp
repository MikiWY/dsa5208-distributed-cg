#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "distributed_vector_ops.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

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

    const double result = distributed_dot_product(
        a_local, b_local, MPI_COMM_WORLD
    );

    const double expected = 20.0;

    const int local_pass =
        std::abs(result - expected) <= 1e-12 ? 1 : 0;

    int global_pass = 0;

    MPI_Allreduce(
        &local_pass,
        &global_pass,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "MPI ranks: " << size << "\n";
        std::cout << "Global dot product: " << result << "\n";

        if (global_pass == 1) {
            std::cout << "PASS: Distributed dot product\n";
        } else {
            std::cerr << "FAIL: Distributed dot product\n";
        }
    }

    MPI_Finalize();

    return global_pass == 1 ? 0 : 1;
}
