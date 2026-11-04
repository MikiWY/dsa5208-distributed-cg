#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "gpu_spmv.hpp"
#include "matrix.hpp"

int main() {
    int device_count = 0;

    cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status != cudaSuccess || device_count == 0) {
        std::cerr << "FAIL: No CUDA GPU available\n";
        return 1;
    }

    cudaDeviceProp device{};
    cudaGetDeviceProperties(&device, 0);

    std::cout << "CUDA device: "
              << device.name << "\n";

    for (int n : {3, 8, 32, 64, 128}) {
        const int N = n * n;

        const CSRMatrix A =
            generate_poisson_matrix(n, 0, N);

        std::vector<double> x(N);

        for (int k = 0; k < N; ++k) {
            const int i = k % n + 1;
            const int j = k / n + 1;

            x[k] =
                1.0
                + 0.1 * std::sin(3.0 * i + 5.0 * j)
                + 0.05 * std::cos(7.0 * i - 2.0 * j);
        }

        std::vector<double> reference(N, 0.0);

        for (int r = 0; r < N; ++r) {
            for (int p = A.row_ptr[r];
                 p < A.row_ptr[r + 1]; ++p) {
                reference[r] +=
                    A.values[p] * x[A.col_idx[p]];
            }
        }

        const std::vector<double> gpu_result =
            gpu_csr_spmv(A, x);

        double max_error = 0.0;
        double max_reference = 0.0;

        for (int i = 0; i < N; ++i) {
            max_error = std::max(
                max_error,
                std::abs(gpu_result[i] - reference[i])
            );

            max_reference = std::max(
                max_reference,
                std::abs(reference[i])
            );
        }

        const double relative_error =
            max_error / std::max(1.0, max_reference);

        std::cout << "n=" << n
                  << ", relative error="
                  << relative_error << "\n";

        if (!std::isfinite(relative_error) ||
            relative_error > 1e-12) {
            std::cerr
                << "FAIL: GPU SpMV numerical consistency\n";
            return 1;
        }
    }

    std::cout
        << "PASS: GPU CSR SpMV matches CPU reference\n";

    return 0;
}
