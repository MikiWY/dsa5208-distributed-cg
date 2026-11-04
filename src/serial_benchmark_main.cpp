#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_csv.hpp"
#include "benchmark_version.hpp"
#include "cg.hpp"
#include "matrix.hpp"

int main(int argc, char** argv) {
    int n = 8;
    std::string run_id = "serial-manual";
    std::string timing_output;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--n" && i + 1 < argc) {
            try {
                const std::string value(argv[++i]);
                std::size_t parsed = 0;
                const long long requested =
                    std::stoll(value, &parsed);

                if (parsed != value.size() ||
                    requested < 1 || requested > 4096) {
                    throw std::invalid_argument(
                        "Invalid grid size"
                    );
                }

                n = static_cast<int>(requested);
            } catch (const std::exception&) {
                std::cerr << "Invalid --n value\n";
                return 1;
            }
        } else if (arg == "--run-id" && i + 1 < argc) {
            run_id = argv[++i];
            if (run_id.empty()) return 1;
        } else if (arg == "--timing-output" &&
                   i + 1 < argc) {
            timing_output = argv[++i];
            if (timing_output.empty()) return 1;
        } else {
            std::cerr
                << "Usage: " << argv[0]
                << " [--n GRID_SIZE]"
                << " [--run-id ID]"
                << " [--timing-output PATH]\n";
            return 1;
        }
    }

    try {
        const int N = n * n;

        // Preprocessing is outside the timed region.
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

        // Independent CSR SpMV for RHS construction.
        const auto spmv = [&](const std::vector<double>& x) {
            std::vector<double> y(N, 0.0);

            for (int r = 0; r < N; ++r) {
                for (int p = A.row_ptr[r];
                     p < A.row_ptr[r + 1]; ++p) {
                    y[r] += A.values[p] * x[A.col_idx[p]];
                }
            }

            return y;
        };

        const std::vector<double> b = spmv(x_true);

        SolverOptions options;
        options.tolerance = 1e-8;

        // CG-only timing.
        const auto start =
            std::chrono::steady_clock::now();

        const SolverResult result =
            conjugate_gradient(A, b, options);

        const auto end =
            std::chrono::steady_clock::now();

        const double time_s =
            std::chrono::duration<double>(
                end - start
            ).count();

        // Independent validation, outside timing.
        const std::vector<double> Ax = spmv(result.x);

        double residual_sq = 0.0;
        double b_sq = 0.0;
        double error_sq = 0.0;
        double true_sq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double residual = b[i] - Ax[i];
            const double error = result.x[i] - x_true[i];

            residual_sq += residual * residual;
            b_sq += b[i] * b[i];
            error_sq += error * error;
            true_sq += x_true[i] * x_true[i];
        }

        const double true_residual =
            b_sq > 0.0
                ? std::sqrt(residual_sq / b_sq)
                : 0.0;

        const double relative_error =
            true_sq > 0.0
                ? std::sqrt(error_sq / true_sq)
                : 0.0;

        const bool passed =
            result.converged &&
            result.termination_reason ==
                TerminationReason::Converged &&
            std::isfinite(time_s) &&
            time_s >= 0.0 &&
            std::isfinite(true_residual) &&
            std::isfinite(relative_error) &&
            true_residual <= 1e-8 &&
            relative_error <= 1e-6;

        std::cout << std::setprecision(12);
        std::cout << "Solver: serial\n";
        std::cout << "Grid n: " << n << "\n";
        std::cout << "Iterations: "
                  << result.iterations << "\n";
        std::cout << "CG-only time_s: "
                  << time_s << "\n";
        std::cout << "True relative residual: "
                  << true_residual << "\n";
        std::cout << "Relative solution error: "
                  << relative_error << "\n";

        if (!timing_output.empty()) {
            BenchmarkTimingRecord record;
            record.run_id = run_id;
            record.dim = 2;
            record.n = n;
            record.comm = "none";
            record.ranks = 1;
            record.threads = 1;
            record.iterations = result.iterations;
            record.converged = result.converged;
            record.time_s = time_s;
            record.relative_residual = true_residual;
            record.git_commit = benchmark_git_commit();

            append_benchmark_timing_csv(
                timing_output, record
            );
        }

        std::cout
            << (passed ? "PASS" : "FAIL")
            << ": Serial CG benchmark\n";

        return passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Serial benchmark error: "
                  << e.what() << "\n";
        return 1;
    }
}
