#include "gpu_timing_csv.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const std::string path =
        "/tmp/dsa5208_gpu_timing_test.csv";

    std::remove(path.c_str());

    GpuTimingRecord record;
    record.run_id = "gpu,test-001";
    record.n = 32;
    record.N = 1024;
    record.device = "NVIDIA A40";
    record.preprocessing_s = 0.01;
    record.h2d_s = 0.004;
    record.time_s = 0.02;
    record.d2h_s = 0.002;
    record.iterations = 74;
    record.converged = true;
    record.git_commit = "test-commit";

    append_gpu_timing_csv(path, record);

    record.run_id = "gpu-test-002";
    append_gpu_timing_csv(path, record);

    std::ifstream input(path);
    assert(input.is_open());

    std::string header;
    std::string first;
    std::string second;
    std::string extra;

    assert(static_cast<bool>(std::getline(input, header)));
    assert(static_cast<bool>(std::getline(input, first)));
    assert(static_cast<bool>(std::getline(input, second)));
    assert(!std::getline(input, extra));

    assert(header ==
        "run_id,n,N,device,preprocessing_s,h2d_s,"
        "time_s,d2h_s,iterations,converged,git_commit");

    assert(first.find("\"gpu,test-001\"") == 0);
    assert(second.find("gpu-test-002") == 0);

    const std::string bad_path =
        "/tmp/dsa5208_gpu_bad_schema.csv";

    {
        std::ofstream bad(bad_path);
        bad << "wrong,header\n";
    }

    bool rejected = false;

    try {
        append_gpu_timing_csv(bad_path, record);
    } catch (const std::runtime_error&) {
        rejected = true;
    }

    assert(rejected);

    std::cout
        << "PASS: GPU Timing CSV creation, append, "
        << "escaping and schema validation\n";

    return 0;
}
