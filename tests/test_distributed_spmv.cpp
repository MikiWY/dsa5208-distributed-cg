#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "distributed_spmv.hpp"
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

    // Partition the global matrix rows.
    RowPartition part = partition_rows(N, rank, size);

    // Each rank generates only its owned CSR rows.
    CSRMatrix A = generate_poisson_matrix(
        n,
        part.global_row_start,
        part.local_num_rows
    );

    // Local portion of the global vector x[k] = k + 1.
    std::vector<double> x_local(part.local_num_rows);

    for (int i = 0; i < part.local_num_rows; ++i) {
        x_local[i] =
            static_cast<double>(part.global_row_start + i + 1);
    }

    // Allgatherv receive layout.
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int r = 0; r < size; ++r) {
        RowPartition p = partition_rows(N, r, size);

        recvcounts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    // Distributed SpMV.
    std::vector<double> y_local =
        distributed_spmv_allgatherv(
            A,
            x_local,
            recvcounts,
            displs,
            MPI_COMM_WORLD
        );

    // Independent n=3 reference from C.
    const std::vector<double> expected = {
        -32.0, -16.0, 64.0,
        48.0, 0.0, 112.0,
        256.0, 176.0, 352.0
    };

    double local_max_error = 0.0;

    for (int i = 0; i < part.local_num_rows; ++i) {
        const int global_row = part.global_row_start + i;

        const double error = std::abs(
            y_local[i] - expected[global_row]
        );

        if (error > local_max_error) {
            local_max_error = error;
        }
    }

    // Global maximum error across all ranks.
    double global_max_error = 0.0;

    MPI_Allreduce(
        &local_max_error,
        &global_max_error,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", N=" << N
                  << ", MPI ranks=" << size << "\n";

        std::cout << "Maximum SpMV error: "
                  << global_max_error << "\n";

        if (global_max_error <= 1e-12) {
            std::cout
                << "PASS: Distributed SpMV Allgatherv\n";
        } else {
            std::cerr
                << "FAIL: Distributed SpMV Allgatherv\n";
        }
    }

    MPI_Finalize();

    return global_max_error <= 1e-12 ? 0 : 1;
}
