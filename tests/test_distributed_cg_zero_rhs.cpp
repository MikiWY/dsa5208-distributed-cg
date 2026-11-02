#include <mpi.h>

#include <iostream>
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

    const int n = 3;
    const int N = n * n;

    RowPartition part = partition_rows(N, rank, size);

    CSRMatrix A = generate_poisson_matrix(
        n,
        part.global_row_start,
        part.local_num_rows
    );

    std::vector<double> b_local(part.local_num_rows, 0.0);

    SolverOptions options;

    SolverResult result =
        distributed_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD
        );

    bool local_pass =
        result.converged &&
        result.iterations == 0 &&
        result.recursive_rel_residual == 0.0 &&
        result.termination_reason ==
            TerminationReason::Converged &&
        result.x.size() ==
            static_cast<std::size_t>(part.local_num_rows);

    for (double value : result.x) {
        if (value != 0.0) {
            local_pass = false;
        }
    }

    int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok,
        &global_ok,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "MPI ranks: " << size << "\n";

        if (global_ok == 1) {
            std::cout
                << "PASS: Distributed CG global zero RHS\n";
        } else {
            std::cerr
                << "FAIL: Distributed CG global zero RHS\n";
        }
    }

    MPI_Finalize();

    return global_ok == 1 ? 0 : 1;
}
