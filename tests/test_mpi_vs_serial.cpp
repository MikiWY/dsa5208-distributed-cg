#include <mpi.h>

#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "cg.hpp"
#include "distributed_cg.hpp"
#include "distributed_spmv.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Optional grid size argument (default n=8).
    int n = 8;

    if (argc > 1) {
        try {
            std::size_t parsed = 0;
            const std::string argument(argv[1]);
            n = std::stoi(argument, &parsed);

            if (parsed != argument.size() ||
                n <= 0 || n > 10000) {
                throw std::invalid_argument("Invalid grid size");
            }
        } catch (const std::exception&) {
            if (rank == 0) {
                std::cerr << "Invalid grid size\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    const int N = n * n;

    RowPartition part = partition_rows(N, rank, size);

    CSRMatrix A_local = generate_poisson_matrix(
        n, part.global_row_start, part.local_num_rows
    );

    std::vector<int> counts(size);
    std::vector<int> displs(size);

    for (int r = 0; r < size; ++r) {
        RowPartition p = partition_rows(N, r, size);
        counts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    // C's approved manufactured solution.
    std::vector<double> x_true_local(part.local_num_rows);

    for (int r = 0; r < part.local_num_rows; ++r) {
        int k = part.global_row_start + r;
        int i = k % n + 1;
        int j = k / n + 1;

        x_true_local[r] =
            1.0
            + 0.1 * std::sin(3.0 * i + 5.0 * j)
            + 0.05 * std::cos(7.0 * i - 2.0 * j);
    }

    std::vector<double> b_local =
        distributed_spmv_allgatherv(
            A_local, x_true_local,
            counts, displs, MPI_COMM_WORLD
        );

    SolverOptions options;

    SolverResult mpi_result =
        distributed_conjugate_gradient_allgatherv(
            A_local, b_local, options, MPI_COMM_WORLD
        );

    // Gather the distributed solution and RHS to rank 0.
    std::vector<double> x_mpi;
    std::vector<double> b_global;

    if (rank == 0) {
        x_mpi.resize(N);
        b_global.resize(N);
    }

    MPI_Gatherv(
        mpi_result.x.data(),
        part.local_num_rows,
        MPI_DOUBLE,
        rank == 0 ? x_mpi.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gatherv(
        b_local.data(),
        part.local_num_rows,
        MPI_DOUBLE,
        rank == 0 ? b_global.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    int passed = 1;

    if (rank == 0) {
        CSRMatrix A_serial =
            generate_poisson_matrix(n, 0, N);

        SolverResult serial_result =
            conjugate_gradient(A_serial, b_global, options);

        double difference_sq = 0.0;
        double reference_sq = 0.0;

        for (int i = 0; i < N; ++i) {
            double diff =
                x_mpi[i] - serial_result.x[i];

            difference_sq += diff * diff;
            reference_sq +=
                serial_result.x[i] * serial_result.x[i];
        }

        const double serial_difference =
            std::sqrt(difference_sq / reference_sq);

        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size << "\n";
        std::cout << "Serial iterations: "
                  << serial_result.iterations << "\n";
        std::cout << "MPI iterations: "
                  << mpi_result.iterations << "\n";
        std::cout << "Relative serial difference: "
                  << serial_difference << "\n";

        if (!serial_result.converged ||
            !mpi_result.converged ||
            !std::isfinite(serial_difference) ||
            serial_difference > 1e-8) {
            passed = 0;
        }

        std::cout << (passed ? "PASS" : "FAIL")
                  << ": MPI vs Serial CG\n";
    }

    MPI_Bcast(
        &passed, 1, MPI_INT, 0, MPI_COMM_WORLD
    );

    MPI_Finalize();
    return passed ? 0 : 1;
}
