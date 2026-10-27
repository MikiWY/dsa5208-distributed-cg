#include "matrix.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

CSRMatrix generate_poisson_matrix(
    int n,
    int global_row_start,
    int local_num_rows
) {
    // Validate the grid and ownership range.
    if (n <= 0 ||
        static_cast<long long>(n) * n >
            std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Invalid grid size");
    }

    const int N = n * n;

    if (global_row_start < 0 ||
        local_num_rows < 0 ||
        global_row_start > N ||
        local_num_rows > N - global_row_start) {
        throw std::invalid_argument("Invalid owned row range");
    }

    const double h = 1.0 / (n + 1.0);
    const double scale = 1.0 / (h * h);

    CSRMatrix A;
    A.global_rows = N;
    A.global_row_start = global_row_start;
    A.local_num_rows = local_num_rows;

    A.row_ptr.push_back(0);

    // Generate only the matrix rows owned by this rank.
    for (int r = 0; r < local_num_rows; ++r) {
        const int k = global_row_start + r;

        const int i = k % n;
        const int j = k / n;

        // Collect nonzero entries for this global row.
        std::vector<std::pair<int, double>> entries;

        // Upper neighbor
        if (j > 0) {
            entries.emplace_back(k - n, -scale);
        }

        // Left neighbor
        if (i > 0) {
            entries.emplace_back(k - 1, -scale);
        }

        // Diagonal
        entries.emplace_back(k, 4.0 * scale);

        // Right neighbor
        if (i < n - 1) {
            entries.emplace_back(k + 1, -scale);
        }

        // Lower neighbor
        if (j < n - 1) {
            entries.emplace_back(k + n, -scale);
        }

        // Entries are already in ascending column order.
        for (const auto& entry : entries) {
            A.col_idx.push_back(entry.first);
            A.values.push_back(entry.second);
        }

        A.row_ptr.push_back(
            static_cast<int>(A.values.size())
        );
    }

    return A;
}
