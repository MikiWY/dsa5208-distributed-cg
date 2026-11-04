#pragma once

#include <string>

struct GpuTimingRecord {
    std::string run_id;
    int n = 0;
    int N = 0;
    std::string device;
    double preprocessing_s = 0.0;
    double h2d_s = 0.0;
    double time_s = 0.0;
    double d2h_s = 0.0;
    int iterations = 0;
    bool converged = false;
    std::string git_commit;
};

void append_gpu_timing_csv(
    const std::string& filename,
    const GpuTimingRecord& record
);
