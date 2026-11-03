#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "distributed_spmv.hpp"
#include "halo_exchange.hpp"
#include "halo_spmv.hpp"
#include "matrix.hpp"
#include "partition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 3;

    if (argc > 1) {
        n = std::stoi(argv[1]);
    }

    if (n != 3 && n != 4) {
        if (rank == 0) {
            std::cerr << "Only n=3 and n=4 supported\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int N = n * n;

    const RowPartition part =
        partition_rows(N, rank, size);

    const CSRMatrix A =
        generate_poisson_matrix(
            n,
            part.global_row_start,
            part.local_num_rows
        );

    std::vector<int> counts(size);
    std::vector<int> displs(size);

    for (int r = 0; r < size; ++r) {
        const RowPartition p = partition_rows(N, r, size);
        counts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    // C's independent reference input p[k] = k + 1.
    std::vector<double> x_local(part.local_num_rows);

    for (int i = 0; i < part.local_num_rows; ++i) {
        x_local[i] =
            static_cast<double>(part.global_row_start + i + 1);
    }

    // MPI Allgatherv reference implementation.
    const std::vector<double> y_allgatherv =
        distributed_spmv_allgatherv(
            A, x_local, counts, displs, MPI_COMM_WORLD
        );

    // Blocking Halo implementation.
    const HaloExchangePlan plan =
        build_halo_exchange_plan(A, MPI_COMM_WORLD);

    const std::vector<double> y_halo =
        distributed_spmv_halo(
            A, x_local, plan, MPI_COMM_WORLD
        );

    // Read C's independent CSV reference.
    const std::string filename =
        "tests/reference/poisson_n" +
        std::to_string(n) + "_spmv.csv";

    std::ifstream in(filename);

    bool local_pass = in.is_open();

    std::string line;

    if (local_pass) {
        std::getline(in, line); // CSV header

        for (int k = 0; k < N; ++k) {
            if (!std::getline(in, line)) {
                local_pass = false;
                break;
            }

            const std::size_t first = line.find(',');
            const std::size_t second =
                line.find(',', first + 1);

            if (first == std::string::npos ||
                second == std::string::npos) {
                local_pass = false;
                break;
            }

            const int global_index =
                std::stoi(line.substr(0, first));

            const double expected_x =
                std::stod(
                    line.substr(first + 1, second - first - 1)
                );

            const double expected_y =
                std::stod(line.substr(second + 1));

            if (global_index != k ||
                expected_x != static_cast<double>(k + 1)) {
                local_pass = false;
                break;
            }

            if (k >= part.global_row_start &&
                k < part.global_row_start + part.local_num_rows) {

                const int local_index =
                    k - part.global_row_start;

                if (!std::isfinite(y_halo[local_index]) ||
                    !std::isfinite(y_allgatherv[local_index]) ||
                    std::abs(y_halo[local_index] - expected_y)
                        > 1e-12 ||
                    std::abs(y_allgatherv[local_index] - expected_y)
                        > 1e-12) {
                    local_pass = false;
                }
            }
        }
    }

    const int local_ok = local_pass ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout << "Grid n=" << n
                  << ", MPI ranks=" << size << "\n";

        std::cout
            << (global_ok ? "PASS" : "FAIL")
            << ": Halo and Allgatherv vs independent reference\n";
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
