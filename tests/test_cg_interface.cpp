#include <cmath>
#include <iostream>

#include "cg.hpp"

int main() {
    SolverOptions options;

    SolverResult result;

    // Check default solver configuration.
    if (std::abs(options.tolerance - 1e-8) > 1e-15) {
        std::cerr << "FAIL: Default tolerance\n";
        return 1;
    }

    if (options.max_iterations != 0) {
        std::cerr << "FAIL: Default iteration policy\n";
        return 1;
    }

    // Check default solver result.
    if (result.iterations != 0 ||
        result.converged ||
        result.recursive_rel_residual != 0.0 ||
        result.termination_reason !=
            TerminationReason::MaxIterations) {
        std::cerr << "FAIL: SolverResult defaults\n";
        return 1;
    }

    std::cout << "Default tolerance: "
              << options.tolerance << "\n";

    std::cout << "Default max_iterations setting: "
              << options.max_iterations << "\n";

    std::cout << "PASS: Serial CG interface test\n";

    return 0;
}
