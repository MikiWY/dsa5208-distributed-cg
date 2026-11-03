#include "benchmark_csv.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>

int main() {
    const std::string filename = "/tmp/test_benchmark_timing.csv";

    std::remove(filename.c_str());

    BenchmarkTimingRecord record;
    record.run_id = "test,001";
    record.dim = 2;
    record.n = 32;
    record.comm = "allgatherv";
    record.ranks = 2;
    record.threads = 1;
    record.iterations = 74;
    record.converged = true;
    record.time_s = 0.001761;
    record.relative_residual = 9.96437835741e-9;
    record.git_commit = "test-commit";

    append_benchmark_timing_csv(filename, record);

    record.run_id = "test-002";
    append_benchmark_timing_csv(filename, record);

    std::ifstream in(filename);
    std::vector<std::string> lines;
    std::string line;

    while (std::getline(in, line)) {
        lines.push_back(line);
    }

    if (lines.size() != 3) {
        std::cerr << "FAIL: Expected header and two records\n";
        return 1;
    }

    if (lines[0] !=
        "run_id,dim,n,comm,ranks,threads,"
        "iterations,converged,time_s,compute_s,"
        "vector_comm_s,reduction_s,"
        "relative_residual,git_commit") {
        std::cerr << "FAIL: Incorrect CSV header\n";
        return 1;
    }

    if (lines[1].find("\"test,001\"") != 0) {
        std::cerr << "FAIL: CSV escaping\n";
        return 1;
    }

    if (lines[1].find("0.001761") == std::string::npos) {
        std::cerr << "FAIL: Timing value missing\n";
        return 1;
    }

    std::cout << "PASS: CSV header\n";
    std::cout << "PASS: Two records appended\n";
    std::cout << "PASS: CSV text escaping\n";
    std::cout << "PASS: Timing value recorded\n";

    return 0;
}
