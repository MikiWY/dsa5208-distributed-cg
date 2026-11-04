#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_options.hpp"

BenchmarkOptions parse(
    const std::vector<std::string>& args
) {
    std::vector<std::string> storage = args;
    std::vector<char*> argv;

    for (auto& arg : storage) {
        argv.push_back(arg.data());
    }

    return parse_benchmark_options(
        static_cast<int>(argv.size()),
        argv.data()
    );
}

bool rejects(const std::vector<std::string>& args) {
    try {
        parse(args);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::out_of_range&) {
        return true;
    }
}

int main() {
    const auto defaults = parse({"benchmark"});

    assert(defaults.n == 8);
    assert(defaults.solver == "mpi");
    assert(defaults.communication == "allgatherv");
    assert(defaults.tolerance == 1e-8);
    assert(defaults.max_iterations == 0);

    const auto configured = parse({
        "benchmark",
        "--n", "128",
        "--solver", "hybrid",
        "--comm", "halo",
        "--tol", "1e-10",
        "--max-iterations", "500",
        "--run-id", "test-001",
        "--output", "correctness.csv",
        "--timing-output", "timing.csv"
    });

    assert(configured.n == 128);
    assert(configured.solver == "hybrid");
    assert(configured.communication == "halo");
    assert(std::abs(configured.tolerance - 1e-10) < 1e-20);
    assert(configured.max_iterations == 500);
    assert(configured.run_id == "test-001");
    assert(configured.correctness_output == "correctness.csv");
    assert(configured.timing_output == "timing.csv");

    assert(rejects({"benchmark", "--n", "0"}));
    assert(rejects({"benchmark", "--n", "4097"}));
    assert(rejects({"benchmark", "--n", "abc"}));
    assert(rejects({"benchmark", "--n", "8abc"}));
    assert(rejects({"benchmark", "--tol", "0"}));
    assert(rejects({"benchmark", "--tol", "-1"}));
    assert(rejects({"benchmark", "--tol", "nan"}));
    assert(rejects({"benchmark", "--tol", "inf"}));
    assert(rejects({"benchmark", "--max-iterations", "-1"}));
    assert(rejects({"benchmark", "--max-iterations", "2147483648"}));
    assert(rejects({"benchmark", "--solver", "serial"}));
    assert(rejects({"benchmark", "--comm", "invalid"}));
    assert(rejects({"benchmark", "--unknown", "value"}));
    assert(rejects({"benchmark", "--n"}));

    std::cout
        << "PASS: Benchmark CLI option parsing\n";

    return 0;
}
