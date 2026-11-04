#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "correctness_csv.hpp"

int main() {
    const std::string path =
        "/tmp/dsa5208_correctness_csv_test.csv";

    std::remove(path.c_str());

    CorrectnessRecord record;
    record.run_id = "test,001";
    record.n = 32;
    record.N = 1024;
    record.solver = "mpi";
    record.comm = "halo";
    record.ranks = 2;
    record.threads = 1;
    record.iterations = 74;
    record.converged = true;
    record.recursive_rel_residual = 9.9e-9;
    record.true_rel_residual = 9.9e-9;
    record.relative_solution_error = 2.1e-8;
    record.passed = true;
    record.git_commit = "test-commit";

    append_correctness_csv(path, record);

    record.run_id = "test-002";
    append_correctness_csv(path, record);

    std::ifstream input(path);
    assert(input.is_open());

    std::string header;
    std::string first;
    std::string second;

    assert(static_cast<bool>(std::getline(input, header)));
    assert(static_cast<bool>(std::getline(input, first)));
    assert(static_cast<bool>(std::getline(input, second)));

    std::string extra;
    assert(!std::getline(input, extra));

    assert(header ==
        "run_id,dim,n,N,solver,comm,ranks,threads,"
        "iterations,converged,recursive_rel_residual,"
        "true_rel_residual,relative_solution_error,"
        "pass,git_commit");

    assert(first.find("\"test,001\"") == 0);
    assert(second.find("test-002") == 0);

    const std::string bad_path =
        "/tmp/dsa5208_bad_correctness_schema.csv";

    {
        std::ofstream bad(bad_path);
        bad << "wrong,header\n";
    }

    bool rejected = false;

    try {
        append_correctness_csv(bad_path, record);
    } catch (const std::runtime_error&) {
        rejected = true;
    }

    assert(rejected);

    std::cout
        << "PASS: Correctness CSV creation, append, "
        << "escaping and schema validation\n";
}
