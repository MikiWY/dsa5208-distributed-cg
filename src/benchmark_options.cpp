#include "benchmark_options.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

long long parse_integer(
    const std::string& text
) {
    std::size_t pos = 0;
    const long long value = std::stoll(text, &pos);

    if (pos != text.size()) {
        throw std::invalid_argument("Invalid integer: " + text);
    }

    return value;
}

double parse_positive_double(
    const std::string& text
) {
    std::size_t pos = 0;
    const double value = std::stod(text, &pos);

    if (pos != text.size() ||
        !std::isfinite(value) ||
        value <= 0.0) {
        throw std::invalid_argument(
            "Invalid positive number: " + text
        );
    }

    return value;
}

} // namespace

BenchmarkOptions parse_benchmark_options(
    int argc,
    char** argv
) {
    BenchmarkOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (i + 1 >= argc) {
            throw std::invalid_argument(
                "Missing value for: " + arg
            );
        }

        const std::string value(argv[++i]);

        if (arg == "--n") {
            const long long n = parse_integer(value);

            if (n < 1 || n > 4096) {
                throw std::invalid_argument(
                    "Grid size must be between 1 and 4096"
                );
            }

            options.n = static_cast<int>(n);

        } else if (arg == "--solver") {
            if (value != "mpi" && value != "hybrid") {
                throw std::invalid_argument(
                    "Invalid solver: " + value
                );
            }

            options.solver = value;

        } else if (arg == "--comm") {
            if (value != "allgatherv" && value != "halo") {
                throw std::invalid_argument(
                    "Invalid communication: " + value
                );
            }

            options.communication = value;

        } else if (arg == "--tol") {
            options.tolerance =
                parse_positive_double(value);

        } else if (arg == "--max-iterations") {
            const long long iterations =
                parse_integer(value);

            if (iterations < 0 ||
                iterations >
                    std::numeric_limits<int>::max()) {
                throw std::invalid_argument(
                    "Invalid maximum iteration count"
                );
            }

            options.max_iterations =
                static_cast<int>(iterations);

        } else if (arg == "--run-id") {
            if (value.empty()) {
                throw std::invalid_argument(
                    "Run ID cannot be empty"
                );
            }

            options.run_id = value;

        } else if (arg == "--output") {
            if (value.empty()) {
                throw std::invalid_argument(
                    "Correctness output cannot be empty"
                );
            }

            options.correctness_output = value;

        } else if (arg == "--timing-output") {
            if (value.empty()) {
                throw std::invalid_argument(
                    "Timing output cannot be empty"
                );
            }

            options.timing_output = value;

        } else {
            throw std::invalid_argument(
                "Unknown argument: " + arg
            );
        }
    }

    return options;
}
