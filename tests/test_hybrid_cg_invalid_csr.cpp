#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "hybrid_cg.hpp"
#include "matrix.hpp"
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

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "Test requires exactly 2 MPI ranks\n";
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

    // Deliberately corrupt CSR data on Rank 1 only.
    if (rank == 1) {
        A.row_ptr[0] = 1;
    }

    SolverOptions options;
    bool caught = false;

    try {
        hybrid_conjugate_gradient_allgatherv(
            A, b_local, options, MPI_COMM_WORLD
        );
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    // Verify that every rank caught the exception.
    int local_ok =
        (caught && provided >= MPI_THREAD_FUNNELED) ? 1 : 0;

    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout
            << (global_ok ? "PASS" : "FAIL")
            << ": Invalid CSR rejected on all MPI ranks\n";
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}
