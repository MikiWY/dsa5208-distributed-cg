#include <algorithm>
#include <iostream>
#include <vector>

#include "halo_exchange.hpp"
#include "matrix.hpp"
#include "partition.hpp"

bool check_case(
    int n,
    int rank,
    int num_ranks,
    const std::vector<int>& expected
) {
    const int N = n * n;

    const RowPartition part =
        partition_rows(N, rank, num_ranks);

    const CSRMatrix A =
        generate_poisson_matrix(
            n,
            part.global_row_start,
            part.local_num_rows
        );

    const std::vector<int> ghosts =
        discover_ghost_indices(A);

    if (ghosts != expected) {
        std::cerr
            << "FAIL: n=" << n
            << ", rank=" << rank
            << ", ranks=" << num_ranks << "\n";

        std::cerr << "Expected:";
        for (int x : expected) {
            std::cerr << " " << x;
        }

        std::cerr << "\nActual:";
        for (int x : ghosts) {
            std::cerr << " " << x;
        }

        std::cerr << "\n";
        return false;
    }

    std::cout
        << "PASS: n=" << n
        << ", rank=" << rank
        << ", ranks=" << num_ranks
        << ", ghost count=" << ghosts.size()
        << "\n";

    return true;
}

int main() {
    bool ok = true;

    // n=3, N=9, 3 ranks, 3 rows per rank.
    // Rank 0 owns rows 0,1,2 and needs rows 3,4,5.
    ok = check_case(
        3, 0, 3, {3, 4, 5}
    ) && ok;

    // Rank 1 owns rows 3,4,5 and needs both adjacent blocks.
    ok = check_case(
        3, 1, 3, {0, 1, 2, 6, 7, 8}
    ) && ok;

    // Rank 2 owns rows 6,7,8.
    ok = check_case(
        3, 2, 3, {3, 4, 5}
    ) && ok;

    // Uneven partition: n=3, N=9, 2 ranks.
    // Rank 0 owns rows 0..4.
    ok = check_case(
        3, 0, 2, {5, 6, 7}
    ) && ok;

    // Rank 1 owns rows 5..8.
    // Remote dependencies are columns 2, 3 and 4.
    ok = check_case(
        3, 1, 2, {2, 3, 4}
    ) && ok;

    // Zero-row rank: n=2, N=4, 5 ranks.
    ok = check_case(
        2, 4, 5, {}
    ) && ok;

    if (!ok) {
        return 1;
    }

    std::cout
        << "PASS: All Halo ghost discovery tests\n";

    return 0;
}
