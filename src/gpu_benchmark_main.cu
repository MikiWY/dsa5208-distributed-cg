#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_csv.hpp"
#include "benchmark_version.hpp"
#include "correctness_csv.hpp"
#include "gpu_benchmark.hpp"
#include "gpu_timing_csv.hpp"
#include "matrix.hpp"

namespace {

struct Options {
    int n = 32;
    double tolerance = 1e-8;
    int max_iterations = 0;
    std::string run_id = "gpu-manual";
    std::string correctness_output;
    std::string timing_output;
    std::string gpu_timing_output;
};

int parse_int(const std::string& s) {
    std::size_t pos = 0;
    long long value = std::stoll(s, &pos);

    if (pos != s.size() ||
        value < 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Invalid integer: " + s);
    }

    return static_cast<int>(value);
}

double parse_double(const std::string& s) {
    std::size_t pos = 0;
    double value = std::stod(s, &pos);

    if (pos != s.size() ||
        !std::isfinite(value) ||
        value <= 0.0) {
        throw std::invalid_argument("Invalid tolerance");
    }

    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (i + 1 >= argc) {
            throw std::invalid_argument(
                "Missing value for " + arg
            );
        }

        std::string value = argv[++i];

        if (arg == "--n") {
            options.n = parse_int(value);
            if (options.n < 1 || options.n > 4096) {
                throw std::invalid_argument(
                    "Grid n must be between 1 and 4096"
                );
            }
        } else if (arg == "--tol") {
            options.tolerance = parse_double(value);
        } else if (arg == "--max-iterations") {
            options.max_iterations = parse_int(value);
        } else if (arg == "--run-id") {
            if (value.empty()) {
                throw std::invalid_argument("Empty run ID");
            }
            options.run_id = value;
        } else if (arg == "--output") {
            if (value.empty()) {
                throw std::invalid_argument("Empty output path");
            }
            options.correctness_output = value;
        } else if (arg == "--timing-output") {
            if (value.empty()) {
                throw std::invalid_argument("Empty timing path");
            }
            options.timing_output = value;
        } else if (arg == "--gpu-timing-output") {
            if (value.empty()) {
                throw std::invalid_argument("Empty GPU timing path");
            }
            options.gpu_timing_output = value;
        } else {
            throw std::invalid_argument(
                "Unknown argument: " + arg
            );
        }
    }

    return options;
}

std::vector<double> spmv(
    const CSRMatrix& A,
    const std::vector<double>& x
) {
    std::vector<double> y(A.global_rows, 0.0);

    for (int r = 0; r < A.local_num_rows; ++r) {
        for (int k = A.row_ptr[r];
             k < A.row_ptr[r + 1]; ++k) {
            y[r] += A.values[k] * x[A.col_idx[k]];
        }
    }

    return y;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options cli = parse_options(argc, argv);

        int count = 0;
        cudaError_t status = cudaGetDeviceCount(&count);

        if (status != cudaSuccess || count == 0) {
            throw std::runtime_error(
                "No CUDA GPU available"
            );
        }

        cudaDeviceProp device{};
        status = cudaGetDeviceProperties(&device, 0);

        if (status != cudaSuccess) {
            throw std::runtime_error(
                "Cannot query CUDA device"
            );
        }

        const int n = cli.n;
        const int N = n * n;

        const CSRMatrix A =
            generate_poisson_matrix(n, 0, N);

        std::vector<double> x_true(N);

        for (int k = 0; k < N; ++k) {
            const int i = k % n + 1;
            const int j = k / n + 1;

            x_true[k] =
                1.0
                + 0.1 * std::sin(3.0 * i + 5.0 * j)
                + 0.05 * std::cos(7.0 * i - 2.0 * j);
        }

        const auto b = spmv(A, x_true);

        SolverOptions options;
        options.tolerance = cli.tolerance;
        options.max_iterations = cli.max_iterations;

        GpuBenchmarkTiming setup;
        GpuCgWorkspace workspace(A, b, &setup);

        GpuBenchmarkTiming timing;
        const SolverResult result =
            workspace.solve(options, &timing);

        const auto Ax = spmv(A, result.x);

        double residual_sq = 0.0;
        double b_sq = 0.0;
        double error_sq = 0.0;
        double true_sq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double r = b[i] - Ax[i];
            const double e = result.x[i] - x_true[i];

            residual_sq += r * r;
            b_sq += b[i] * b[i];
            error_sq += e * e;
            true_sq += x_true[i] * x_true[i];
        }

        const double true_residual =
            std::sqrt(residual_sq / b_sq);

        const double relative_error =
            std::sqrt(error_sq / true_sq);

        const bool passed =
            result.converged &&
            std::isfinite(true_residual) &&
            std::isfinite(relative_error) &&
            true_residual <= 1e-8 &&
            relative_error <= 1e-6;

        std::cout << std::setprecision(12)
                  << "GPU: " << device.name << "\n"
                  << "Grid n: " << n << "\n"
                  << "Iterations: " << result.iterations << "\n"
                  << "CG-only time_s: " << timing.time_s << "\n"
                  << "Preprocessing_s: "
                  << setup.preprocessing_s << "\n"
                  << "H2D_s: " << setup.h2d_s << "\n"
                  << "D2H_s: " << timing.d2h_s << "\n"
                  << "True relative residual: "
                  << true_residual << "\n"
                  << "Relative solution error: "
                  << relative_error << "\n";

        const std::string commit =
            benchmark_git_commit();

        if (!cli.correctness_output.empty()) {
            CorrectnessRecord record;
            record.run_id = cli.run_id;
            record.dim = 2;
            record.n = n;
            record.N = N;
            record.solver = "gpu";
            record.comm = "none";
            record.ranks = 1;
            record.threads = 1;
            record.iterations = result.iterations;
            record.converged = result.converged;
            record.recursive_rel_residual =
                result.recursive_rel_residual;
            record.true_rel_residual = true_residual;
            record.relative_solution_error =
                relative_error;
            record.passed = passed;
            record.git_commit = commit;

            append_correctness_csv(
                cli.correctness_output, record
            );
        }

        if (!cli.timing_output.empty()) {
            BenchmarkTimingRecord record;
            record.run_id = cli.run_id;
            record.dim = 2;
            record.n = n;
            record.comm = "gpu";
            record.ranks = 1;
            record.threads = 1;
            record.iterations = result.iterations;
            record.converged = result.converged;
            record.time_s = timing.time_s;
            record.relative_residual = true_residual;
            record.git_commit = commit;

            append_benchmark_timing_csv(
                cli.timing_output, record
            );
        }

        if (!cli.gpu_timing_output.empty()) {
            GpuTimingRecord record;
            record.run_id = cli.run_id;
            record.n = n;
            record.N = N;
            record.device = device.name;
            record.preprocessing_s = setup.preprocessing_s;
            record.h2d_s = setup.h2d_s;
            record.time_s = timing.time_s;
            record.d2h_s = timing.d2h_s;
            record.iterations = result.iterations;
            record.converged = result.converged;
            record.git_commit = commit;

            append_gpu_timing_csv(
                cli.gpu_timing_output, record
            );
        }

        std::cout
            << (passed ? "PASS" : "FAIL")
            << ": GPU CG benchmark\n";

        return passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "GPU benchmark error: "
                  << e.what() << "\n";
        return 1;
    }
}
