#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "distributed_cg.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires 2 MPI ranks\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int n = 3;
    const int N = n * n;

    RowPartition part = partition_rows(N, rank, size);

    CSRMatrix A = generate_poisson_matrix(
        n, part.global_row_start, part.local_num_rows
    );

    std::vector<double> b_local(part.local_num_rows, 1.0);

    SolverOptions options;
    options.tolerance = (rank == 0) ? 1e-8 : 1e-4;

    bool caught = false;

    try {
        distributed_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD
        );
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    int local_ok = caught ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        if (global_ok == 1) {
            std::cout
                << "PASS: Inconsistent MPI tolerance rejected on all ranks\n";
        } else {
            std::cerr
                << "FAIL: Inconsistent MPI tolerance not rejected\n";
        }
    }

    // Test 2: Inconsistent maximum iteration limits.
    SolverOptions options2;
    options2.max_iterations = (rank == 0) ? 100 : 200;

    bool caught_iterations = false;

    try {
        distributed_conjugate_gradient_allgatherv(
            A, b_local, options2, MPI_COMM_WORLD
        );
    } catch (const std::invalid_argument&) {
        caught_iterations = true;
    }

    int local_iterations_ok = caught_iterations ? 1 : 0;
    int global_iterations_ok = 0;

    MPI_Allreduce(
        &local_iterations_ok,
        &global_iterations_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        if (global_iterations_ok == 1) {
            std::cout
                << "PASS: Inconsistent MPI max_iterations rejected on all ranks\n";
        } else {
            std::cerr
                << "FAIL: Inconsistent MPI max_iterations not rejected\n";
        }
    }

    MPI_Finalize();
    return (global_ok && global_iterations_ok) ? 0 : 1;
}
