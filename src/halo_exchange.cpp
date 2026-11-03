#include "halo_exchange.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

// Discover remote vector entries required by local CSR rows.
std::vector<int> discover_ghost_indices(
    const CSRMatrix& A
) {
    if (A.global_rows < 0 ||
        A.global_row_start < 0 ||
        A.local_num_rows < 0 ||
        A.global_row_start > A.global_rows ||
        A.local_num_rows >
            A.global_rows - A.global_row_start) {
        throw std::invalid_argument(
            "Invalid Halo matrix dimensions"
        );
    }

    if (A.row_ptr.size() !=
            static_cast<std::size_t>(A.local_num_rows) + 1 ||
        A.col_idx.size() != A.values.size() ||
        A.row_ptr.empty() ||
        A.row_ptr.front() != 0 ||
        A.row_ptr.back() < 0 ||
        static_cast<std::size_t>(A.row_ptr.back()) !=
            A.col_idx.size()) {
        throw std::invalid_argument(
            "Invalid Halo CSR structure"
        );
    }

    const int nnz = A.row_ptr.back();

    for (int r = 0; r < A.local_num_rows; ++r) {
        if (A.row_ptr[r] < 0 ||
            A.row_ptr[r] > A.row_ptr[r + 1] ||
            A.row_ptr[r + 1] > nnz) {
            throw std::invalid_argument(
                "Invalid Halo CSR row pointers"
            );
        }
    }

    std::vector<int> ghosts;

    const int local_start = A.global_row_start;
    const int local_end =
        A.global_row_start + A.local_num_rows;

    for (int col : A.col_idx) {
        if (col < 0 || col >= A.global_rows) {
            throw std::invalid_argument(
                "Invalid Halo CSR column index"
            );
        }

        if (col < local_start || col >= local_end) {
            ghosts.push_back(col);
        }
    }

    std::sort(ghosts.begin(), ghosts.end());

    ghosts.erase(
        std::unique(ghosts.begin(), ghosts.end()),
        ghosts.end()
    );

    return ghosts;
}

// Find the owner of a global index under contiguous
// balanced row-block partitioning.
int find_owner_rank(
    int global_index,
    int global_rows,
    int num_ranks
) {
    if (global_rows <= 0 ||
        num_ranks <= 0 ||
        global_index < 0 ||
        global_index >= global_rows) {
        throw std::invalid_argument(
            "Invalid Halo owner lookup parameters"
        );
    }

    const int base = global_rows / num_ranks;
    const int remainder = global_rows % num_ranks;

    // First 'remainder' ranks own base + 1 rows.
    const long long larger_rows =
        static_cast<long long>(remainder) * (base + 1LL);

    if (global_index < larger_rows) {
        return static_cast<int>(
            global_index / (base + 1)
        );
    }

    // Remaining nonempty ranks own base rows.
    // If base == 0, all valid indices were handled above.
    return remainder +
        static_cast<int>(
            (global_index - larger_rows) / base
        );
}

// Build the communication counts and request layout.
// The actual index-request exchange is added next.
HaloExchangePlan build_halo_exchange_plan(
    const CSRMatrix& A,
    MPI_Comm comm
) {
    int rank = 0;
    int size = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    HaloExchangePlan plan;

    plan.global_rows = A.global_rows;
    plan.global_row_start = A.global_row_start;
    plan.local_num_rows = A.local_num_rows;

    // Every rank validates its local CSR before collectives.
    // Invalid inputs must be rejected collectively.
    bool local_valid =
        A.global_rows > 0 &&
        A.global_row_start >= 0 &&
        A.local_num_rows >= 0 &&
        A.global_row_start <= A.global_rows &&
        A.local_num_rows <= A.global_rows - A.global_row_start &&
        A.row_ptr.size() ==
            static_cast<std::size_t>(A.local_num_rows) + 1 &&
        A.col_idx.size() == A.values.size();

    if (local_valid) {
        local_valid =
            !A.row_ptr.empty() &&
            A.row_ptr.front() == 0 &&
            A.row_ptr.back() >= 0 &&
            static_cast<std::size_t>(A.row_ptr.back()) ==
                A.col_idx.size();
    }

    if (local_valid) {
        for (int r = 0; r < A.local_num_rows; ++r) {
            if (A.row_ptr[r] < 0 ||
                A.row_ptr[r] > A.row_ptr[r + 1] ||
                A.row_ptr[r + 1] > A.row_ptr.back()) {
                local_valid = false;
                break;
            }
        }
    }

    if (local_valid) {
        for (int col : A.col_idx) {
            if (col < 0 || col >= A.global_rows) {
                local_valid = false;
                break;
            }
        }
    }

    // Validate global dimension and expected row ownership.
    int min_N = 0;
    int max_N = 0;

    MPI_Allreduce(
        &A.global_rows, &min_N,
        1, MPI_INT, MPI_MIN, comm
    );

    MPI_Allreduce(
        &A.global_rows, &max_N,
        1, MPI_INT, MPI_MAX, comm
    );

    if (min_N != max_N || min_N <= 0) {
        local_valid = false;
    }

    if (local_valid) {
        const int base = A.global_rows / size;
        const int remainder = A.global_rows % size;

        const int expected_count =
            base + (rank < remainder ? 1 : 0);

        const int expected_start =
            rank * base + std::min(rank, remainder);

        if (A.local_num_rows != expected_count ||
            A.global_row_start != expected_start) {
            local_valid = false;
        }
    }

    int local_ok = local_valid ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, comm
    );

    if (!global_ok) {
        throw std::invalid_argument(
            "Invalid Halo Exchange Plan inputs"
        );
    }

    // Discover remote global indices.
    plan.ghost_indices = discover_ghost_indices(A);

    plan.recv_counts.assign(size, 0);
    plan.send_counts.assign(size, 0);
    plan.recv_displs.assign(size, 0);
    plan.send_displs.assign(size, 0);

    // Count requests by owner rank.
    for (int global_index : plan.ghost_indices) {
        const int owner =
            find_owner_rank(
                global_index, A.global_rows, size
            );

        ++plan.recv_counts[owner];
    }

    // Exchange request counts.
    MPI_Alltoall(
        plan.recv_counts.data(), 1, MPI_INT,
        plan.send_counts.data(), 1, MPI_INT,
        comm
    );

    // Compute prefix-sum displacements.
    int total_recv = 0;
    int total_send = 0;

    for (int r = 0; r < size; ++r) {
        plan.recv_displs[r] = total_recv;
        plan.send_displs[r] = total_send;

        total_recv += plan.recv_counts[r];
        total_send += plan.send_counts[r];
    }

    // Group requested global indices by owner rank.
    plan.recv_global_indices.resize(total_recv);

    std::vector<int> cursor = plan.recv_displs;

    for (int global_index : plan.ghost_indices) {
        const int owner =
            find_owner_rank(
                global_index, A.global_rows, size
            );

        plan.recv_global_indices[cursor[owner]++] =
            global_index;
    }

    // Exchange requested global indices.
    //
    // Each rank sends its requested indices to their owners.
    // Each owner receives the indices requested by other ranks.
    std::vector<int> requested_indices(total_send);

    MPI_Alltoallv(
        plan.recv_global_indices.data(),
        plan.recv_counts.data(),
        plan.recv_displs.data(),
        MPI_INT,
        requested_indices.data(),
        plan.send_counts.data(),
        plan.send_displs.data(),
        MPI_INT,
        comm
    );

    // Convert requested global indices to local offsets.
    plan.send_local_indices.resize(total_send);

    bool requests_valid = true;

    const int local_start = A.global_row_start;
    const int local_end =
        local_start + A.local_num_rows;

    for (int i = 0; i < total_send; ++i) {
        const int global_index = requested_indices[i];

        if (global_index < local_start ||
            global_index >= local_end) {
            requests_valid = false;
            break;
        }

        plan.send_local_indices[i] =
            global_index - local_start;
    }

    // All ranks must agree that requests are valid.
    int request_ok = requests_valid ? 1 : 0;
    int global_request_ok = 0;

    MPI_Allreduce(
        &request_ok,
        &global_request_ok,
        1,
        MPI_INT,
        MPI_MIN,
        comm
    );

    if (!global_request_ok) {
        throw std::invalid_argument(
            "Invalid Halo index request"
        );
    }

    return plan;
}

// Exchange halo vector values using deterministic blocking
// pairwise MPI_Sendrecv communication.
std::vector<double> exchange_halo_values_blocking(
    const HaloExchangePlan& plan,
    const std::vector<double>& x_local,
    MPI_Comm comm
) {
    int rank = 0;
    int size = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Validate local metadata before communication.
    bool local_valid =
        plan.global_rows > 0 &&
        plan.global_row_start >= 0 &&
        plan.local_num_rows >= 0 &&
        plan.global_row_start <= plan.global_rows &&
        plan.local_num_rows <=
            plan.global_rows - plan.global_row_start &&
        x_local.size() ==
            static_cast<std::size_t>(plan.local_num_rows) &&
        plan.send_counts.size() ==
            static_cast<std::size_t>(size) &&
        plan.recv_counts.size() ==
            static_cast<std::size_t>(size) &&
        plan.send_displs.size() ==
            static_cast<std::size_t>(size) &&
        plan.recv_displs.size() ==
            static_cast<std::size_t>(size);

    // Validate contiguous communication segments.
    if (local_valid) {
        std::size_t send_total = 0;
        std::size_t recv_total = 0;

        for (int peer = 0; peer < size; ++peer) {
            if (plan.send_counts[peer] < 0 ||
                plan.recv_counts[peer] < 0 ||
                plan.send_displs[peer] < 0 ||
                plan.recv_displs[peer] < 0 ||
                static_cast<std::size_t>(
                    plan.send_displs[peer]) != send_total ||
                static_cast<std::size_t>(
                    plan.recv_displs[peer]) != recv_total) {
                local_valid = false;
                break;
            }

            send_total +=
                static_cast<std::size_t>(
                    plan.send_counts[peer]);

            recv_total +=
                static_cast<std::size_t>(
                    plan.recv_counts[peer]);
        }

        if (local_valid) {
            local_valid =
                send_total == plan.send_local_indices.size() &&
                recv_total == plan.recv_global_indices.size();
        }
    }

    if (local_valid) {
        for (int index : plan.send_local_indices) {
            if (index < 0 ||
                index >= plan.local_num_rows) {
                local_valid = false;
                break;
            }
        }
    }

    // All ranks must agree before entering Sendrecv.
    const int local_ok = local_valid ? 1 : 0;
    int global_ok = 0;

    MPI_Allreduce(
        &local_ok, &global_ok,
        1, MPI_INT, MPI_MIN, comm
    );

    if (!global_ok) {
        throw std::invalid_argument(
            "Invalid blocking Halo Exchange inputs"
        );
    }

    // Pack values requested by remote ranks.
    std::vector<double> send_values(
        plan.send_local_indices.size()
    );

    for (std::size_t i = 0;
         i < plan.send_local_indices.size(); ++i) {
        send_values[i] =
            x_local[plan.send_local_indices[i]];
    }

    std::vector<double> ghost_values(
        plan.recv_global_indices.size(), 0.0
    );

    constexpr int halo_tag = 170;

    // All ranks traverse pairs in the same global order.
    // Only the two participating ranks communicate.
    for (int low = 0; low < size; ++low) {
        for (int high = low + 1; high < size; ++high) {
            if (rank != low && rank != high) {
                continue;
            }

            const int peer = (rank == low) ? high : low;

            const int send_count = plan.send_counts[peer];
            const int recv_count = plan.recv_counts[peer];

            const double* send_ptr =
                send_count > 0
                    ? send_values.data() +
                        plan.send_displs[peer]
                    : nullptr;

            double* recv_ptr =
                recv_count > 0
                    ? ghost_values.data() +
                        plan.recv_displs[peer]
                    : nullptr;

            MPI_Sendrecv(
                send_ptr,
                send_count,
                MPI_DOUBLE,
                peer,
                halo_tag,
                recv_ptr,
                recv_count,
                MPI_DOUBLE,
                peer,
                halo_tag,
                comm,
                MPI_STATUS_IGNORE
            );
        }
    }

    return ghost_values;
}
