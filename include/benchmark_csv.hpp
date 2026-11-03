#pragma once

#include <string>

// One benchmark run, recorded by rank 0 only.
// Optional timing fields remain empty until measured.
struct BenchmarkTimingRecord {
    std::string run_id;
    int dim = 2;
    int n = 0;
    std::string comm = "allgatherv";
    int ranks = 1;
    int threads = 1;
    int iterations = 0;
    bool converged = false;

    double time_s = 0.0;

    std::string compute_s;
    std::string vector_comm_s;
    std::string reduction_s;

    double relative_residual = 0.0;
    std::string git_commit;
};

// Append one record to a timing CSV.
// Create the file and header if it does not exist.
// The parent directory must already exist.
// Call only from rank 0, outside the timed region.
void append_benchmark_timing_csv(
    const std::string& filename,
    const BenchmarkTimingRecord& record
);
