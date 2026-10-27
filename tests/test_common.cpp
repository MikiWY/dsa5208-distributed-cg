#include <iostream>
#include "common.hpp"

int main() {
    CSRMatrix A;

    A.global_rows = 4;
    A.global_row_start = 0;
    A.local_num_rows = 4;

    A.row_ptr = {0, 2, 4, 6, 8};
    A.col_idx = {0, 1, 0, 1, 2, 3, 2, 3};
    A.values = {2.0, -1.0, -1.0, 2.0,
                2.0, -1.0, -1.0, 2.0};

    std::cout << "Global rows: " << A.global_rows << "\n";
    std::cout << "Local rows: " << A.local_num_rows << "\n";
    std::cout << "Nonzero entries: " << A.values.size() << "\n";

    if (A.row_ptr.size() !=
        static_cast<std::size_t>(A.local_num_rows + 1)) {
        std::cerr << "FAIL: Invalid row_ptr size\n";
        return 1;
    }

    if (A.col_idx.size() != A.values.size()) {
        std::cerr << "FAIL: CSR array sizes differ\n";
        return 1;
    }

    std::cout << "PASS: CSR structure test\n";
    return 0;
}
