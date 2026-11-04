#include "correctness_csv.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace {

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";

    for (char c : value) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }

    escaped += '"';
    return escaped;
}

const char* header =
    "run_id,dim,n,N,solver,comm,ranks,threads,"
    "iterations,converged,recursive_rel_residual,"
    "true_rel_residual,relative_solution_error,"
    "pass,git_commit";

} // namespace

void append_correctness_csv(
    const std::string& filename,
    const CorrectnessRecord& record
) {
    if (filename.empty()) {
        throw std::invalid_argument(
            "Correctness CSV filename cannot be empty"
        );
    }

    std::ifstream existing(filename, std::ios::binary);
    bool needs_header = true;

    if (existing.is_open()) {
        std::string first_line;
        std::getline(existing, first_line);

        if (!first_line.empty() &&
            first_line.back() == '\r') {
            first_line.pop_back();
        }

        if (!first_line.empty() && first_line != header) {
            throw std::runtime_error(
                "Existing correctness CSV schema mismatch"
            );
        }

        needs_header = first_line.empty();
    }

    std::ofstream out(filename, std::ios::app);

    if (!out) {
        throw std::runtime_error(
            "Cannot open correctness CSV: " + filename
        );
    }

    if (needs_header) {
        out << header << '\n';
    }

    out << std::setprecision(17)
        << csv_escape(record.run_id) << ','
        << record.dim << ','
        << record.n << ','
        << record.N << ','
        << csv_escape(record.solver) << ','
        << csv_escape(record.comm) << ','
        << record.ranks << ','
        << record.threads << ','
        << record.iterations << ','
        << (record.converged ? 1 : 0) << ','
        << record.recursive_rel_residual << ','
        << record.true_rel_residual << ','
        << record.relative_solution_error << ','
        << (record.passed ? 1 : 0) << ','
        << csv_escape(record.git_commit)
        << '\n';

    out.flush();

    if (!out) {
        throw std::runtime_error(
            "Failed to write correctness CSV"
        );
    }
}
