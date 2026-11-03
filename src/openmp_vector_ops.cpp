#include "openmp_vector_ops.hpp"

#include <stdexcept>
#include <vector>

std::vector<double> openmp_csr_spmv(
    const CSRMatrix& A,
    const std::vector<double>& x_global
) {
    // Validate basic dimensions.
    if (A.global_rows <= 0 ||
        A.local_num_rows < 0 ||
        A.global_row_start < 0 ||
        A.global_row_start > A.global_rows ||
        A.local_num_rows > A.global_rows - A.global_row_start ||
        x_global.size() !=
            static_cast<std::size_t>(A.global_rows)) {
        throw std::invalid_argument(
            "Invalid OpenMP SpMV dimensions"
        );
    }

    // Validate CSR structure before entering parallel region.
    if (A.row_ptr.size() !=
            static_cast<std::size_t>(A.local_num_rows) + 1 ||
        A.row_ptr.empty() ||
        A.row_ptr.front() != 0 ||
        A.col_idx.size() != A.values.size() ||
        A.row_ptr.back() < 0 ||
        static_cast<std::size_t>(A.row_ptr.back()) !=
            A.values.size()) {
        throw std::invalid_argument(
            "Invalid OpenMP SpMV CSR structure"
        );
    }

    const int nnz = A.row_ptr.back();

    for (int r = 0; r < A.local_num_rows; ++r) {
        if (A.row_ptr[r] < 0 ||
            A.row_ptr[r] > A.row_ptr[r + 1] ||
            A.row_ptr[r + 1] > nnz) {
            throw std::invalid_argument(
                "Invalid CSR row pointers"
            );
        }
    }

    for (int col : A.col_idx) {
        if (col < 0 || col >= A.global_rows) {
            throw std::invalid_argument(
                "Invalid CSR column index"
            );
        }
    }

    // Each thread computes independent output rows.
    std::vector<double> y_local(A.local_num_rows, 0.0);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < A.local_num_rows; ++r) {
        double sum = 0.0;

        for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
            sum += A.values[p] * x_global[A.col_idx[p]];
        }

        y_local[r] = sum;
    }

    return y_local;
}

// Compute local vector dot product using OpenMP reduction.
double openmp_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local
) {
    if (a_local.size() != b_local.size()) {
        throw std::invalid_argument(
            "OpenMP dot product vector sizes do not match"
        );
    }

    double sum = 0.0;

    const std::size_t N = a_local.size();

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        sum += a_local[i] * b_local[i];
    }

    return sum;
}

// Update y = y + alpha * x using OpenMP.
void openmp_axpy(
    double alpha,
    const std::vector<double>& x,
    std::vector<double>& y
) {
    if (x.size() != y.size()) {
        throw std::invalid_argument(
            "OpenMP AXPY vector sizes do not match"
        );
    }

    const std::size_t N = x.size();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }
}
