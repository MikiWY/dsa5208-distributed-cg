#include <mpi.h>

#include <iostream>
#include <vector>

#include "partition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 10;

    // Determine the rows owned by this MPI rank.
    RowPartition part = partition_rows(N, rank, size);

    // Local vector contains global indices as test values.
    std::vector<double> local(part.local_num_rows);

    for (int i = 0; i < part.local_num_rows; ++i) {
        local[i] = static_cast<double>(
            part.global_row_start + i
        );
    }

    // Build receive counts and displacements.
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int r = 0; r < size; ++r) {
        RowPartition p = partition_rows(N, r, size);

        recvcounts[r] = p.local_num_rows;
        displs[r] = p.global_row_start;
    }

    // Every rank receives the complete vector.
    std::vector<double> global(N, -1.0);

    MPI_Allgatherv(
        local.data(),
        part.local_num_rows,
        MPI_DOUBLE,
        global.data(),
        recvcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    // Check that every rank received the correct vector.
    bool local_pass = true;

    for (int i = 0; i < N; ++i) {
        if (global[i] != static_cast<double>(i)) {
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
        if (global_ok == 1) {
            std::cout
                << "PASS: MPI_Allgatherv with "
                << size << " ranks, N=" << N << "\n";
        } else {
            std::cerr
                << "FAIL: MPI_Allgatherv validation\n";
        }
    }

    MPI_Finalize();

    return global_ok == 1 ? 0 : 1;
}
