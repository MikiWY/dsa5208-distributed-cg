#pragma once

#include <string>

struct BenchmarkOptions {
    int n = 8;

    std::string solver = "mpi";
    std::string communication = "allgatherv";

    double tolerance = 1e-8;
    int max_iterations = 0;

    std::string run_id = "manual";
    std::string correctness_output;
    std::string timing_output;
};

// Parse benchmark CLI arguments.
// Throws std::invalid_argument for unsupported or invalid inputs.
BenchmarkOptions parse_benchmark_options(
    int argc,
    char** argv
);
