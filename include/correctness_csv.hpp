#pragma once

#include <string>

struct CorrectnessRecord {
    std::string run_id;
    int dim = 2;
    int n = 0;
    int N = 0;
    std::string solver;
    std::string comm;
    int ranks = 1;
    int threads = 1;
    int iterations = 0;
    bool converged = false;
    double recursive_rel_residual = 0.0;
    double true_rel_residual = 0.0;
    double relative_solution_error = 0.0;
    bool passed = false;
    std::string git_commit;
};

void append_correctness_csv(
    const std::string& filename,
    const CorrectnessRecord& record
);
