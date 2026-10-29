#include <iostream>
#include <stdexcept>
#include <vector>

#include "partition.hpp"

bool check_partition(
    int N,
    int P,
    const std::vector<int>& expected_counts
) {
    std::vector<int> ownership(N, 0);
    int next_start = 0;

    for (int rank = 0; rank < P; ++rank) {
        RowPartition part = partition_rows(N, rank, P);

        if (part.global_row_start != next_start ||
            part.local_num_rows != expected_counts[rank]) {
            std::cerr << "FAIL: Incorrect partition for rank "
                      << rank << "\n";
            return false;
        }

        for (int r = 0; r < part.local_num_rows; ++r) {
            int global_row = part.global_row_start + r;

            if (global_row < 0 || global_row >= N) {
                std::cerr << "FAIL: Row out of bounds\n";
                return false;
            }

            ownership[global_row]++;
        }

        next_start += part.local_num_rows;
    }

    if (next_start != N) {
        std::cerr << "FAIL: Incomplete row coverage\n";
        return false;
    }

    for (int count : ownership) {
        if (count != 1) {
            std::cerr << "FAIL: Duplicate or missing row\n";
            return false;
        }
    }

    return true;
}

int main() {
    if (!check_partition(9, 3, {3, 3, 3})) {
        return 1;
    }
    std::cout << "PASS: Even partition\n";

    if (!check_partition(10, 3, {4, 3, 3})) {
        return 1;
    }
    std::cout << "PASS: Uneven partition\n";

    if (!check_partition(2, 4, {1, 1, 0, 0})) {
        return 1;
    }
    std::cout << "PASS: Zero-row ranks\n";

    try {
        partition_rows(-1, 0, 3);
        std::cerr << "FAIL: Invalid input accepted\n";
        return 1;
    } catch (const std::invalid_argument&) {
        std::cout << "PASS: Invalid input rejected\n";
    }

    std::cout << "PASS: All row partition tests\n";
    return 0;
}
