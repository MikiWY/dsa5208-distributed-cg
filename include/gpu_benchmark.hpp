#pragma once

#include <memory>
#include <vector>

#include "cg.hpp"
#include "common.hpp"

// GPU benchmark timing, measured in seconds.
struct GpuBenchmarkTiming {
    double preprocessing_s = 0.0;
    double h2d_s = 0.0;
    double time_s = 0.0;
    double d2h_s = 0.0;
};

// Opaque GPU workspace.
//
// Owns device-resident CSR matrix and CG vectors.
// CUDA resources are released automatically.
class GpuCgWorkspace {
public:
    GpuCgWorkspace(
        const CSRMatrix& A,
        const std::vector<double>& b,
        GpuBenchmarkTiming* timing = nullptr
    );

    ~GpuCgWorkspace();

    GpuCgWorkspace(const GpuCgWorkspace&) = delete;
    GpuCgWorkspace& operator=(
        const GpuCgWorkspace&
    ) = delete;

    GpuCgWorkspace(GpuCgWorkspace&&) noexcept;
    GpuCgWorkspace& operator=(
        GpuCgWorkspace&&
    ) noexcept;

    // Solve with already prepared GPU data.
    //
    // The CG-only timing excludes matrix upload,
    // initial allocation and final solution download.
    SolverResult solve(
        const SolverOptions& options,
        GpuBenchmarkTiming* timing = nullptr
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
