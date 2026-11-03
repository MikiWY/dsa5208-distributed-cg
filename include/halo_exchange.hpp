#pragma once

#include <mpi.h>
#include <vector>

#include "common.hpp"

// Communication layout for blocking Halo Exchange.
//
// CSR column indices remain global.
//
// Each rank owns a contiguous range of vector entries.
// Ghost entries represent remotely owned vector values
// required by local CSR matrix rows.
//
// Communication lists are constructed during setup,
// outside the CG iteration loop.

struct HaloExchangePlan {
    int global_rows = 0;
    int global_row_start = 0;
    int local_num_rows = 0;

    // Sorted unique global indices needed from other ranks.
    std::vector<int> ghost_indices;

    // Communication metadata indexed by MPI rank.
    // Counts are numbers of vector elements.
    std::vector<int> send_counts;
    std::vector<int> recv_counts;

    // Prefix-sum displacements indexed by MPI rank.
    // Used to locate each peer's segment in flattened arrays.
    std::vector<int> send_displs;
    std::vector<int> recv_displs;

    // Flattened local offsets to send to remote ranks.
    std::vector<int> send_local_indices;

    // Flattened global indices expected from remote ranks.
    std::vector<int> recv_global_indices;
};

// Build a reusable Halo Exchange communication plan.
//
// A contains locally owned CSR rows with global column indices.
// The row partition follows partition_rows().
//
// All MPI ranks must participate in this setup.
//
// The returned plan is reused across CG iterations.
HaloExchangePlan build_halo_exchange_plan(
    const CSRMatrix& A,
    MPI_Comm comm
);

// Discover sorted unique remote global column indices.
//
// A must have valid CSR metadata and global column indices.
// This function performs no MPI communication.
//
// A column is remote if it is outside the locally owned
// interval [global_row_start, global_row_start + local_num_rows).
//
// Supports ranks with zero locally owned rows.

std::vector<int> discover_ghost_indices(
    const CSRMatrix& A
);

// Find the MPI rank owning a global vector index.
//
// Ownership follows partition_rows().
// Returns a rank in [0, num_ranks).
//
// Requirements:
//   global_rows > 0
//   num_ranks > 0
//   0 <= global_index < global_rows
//
// Throws std::invalid_argument for invalid inputs.
// Performs no MPI communication.

int find_owner_rank(
    int global_index,
    int global_rows,
    int num_ranks
);

// Exchange remote vector values using blocking MPI communication.
//
// x_local contains only the vector entries owned by this rank.
//
// The HaloExchangePlan must already be constructed collectively
// using build_halo_exchange_plan().
//
// Returns ghost values in the exact order of
// plan.recv_global_indices.
//
// This order may differ from plan.ghost_indices.
//
// Communication:
//   Uses MPI_Sendrecv with a deterministic peer order.
//   All ranks must participate in the same exchange.
//
// MPI threading:
//   Must be called by the thread that initialized MPI,
//   outside OpenMP parallel regions.
//
// Timing:
//   Communication performed here belongs to vector_comm_s.
//   Plan construction is preprocessing and should be timed
//   separately from the CG iteration.

std::vector<double> exchange_halo_values_blocking(
    const HaloExchangePlan& plan,
    const std::vector<double>& x_local,
    MPI_Comm comm
);
