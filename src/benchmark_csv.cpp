#include "benchmark_csv.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace {

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") ==
        std::string::npos) {
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

} // namespace

void append_benchmark_timing_csv(
    const std::string& filename,
    const BenchmarkTimingRecord& record
) {
    if (filename.empty()) {
        throw std::invalid_argument(
            "Timing CSV filename cannot be empty"
        );
    }

    // Determine whether the CSV already contains data.
    std::ifstream existing(filename, std::ios::binary);

    bool needs_header = true;

    if (existing.is_open()) {
        existing.seekg(0, std::ios::end);
        needs_header = (existing.tellg() == 0);
    }

    std::ofstream out(filename, std::ios::app);

    if (!out) {
        throw std::runtime_error(
            "Cannot open timing CSV: " + filename
        );
    }

    if (needs_header) {
        out << "run_id,dim,n,comm,ranks,threads,"
            << "iterations,converged,time_s,compute_s,"
            << "vector_comm_s,reduction_s,"
            << "relative_residual,git_commit\n";
    }

    out << std::setprecision(17);

    out << csv_escape(record.run_id) << ','
        << record.dim << ','
        << record.n << ','
        << csv_escape(record.comm) << ','
        << record.ranks << ','
        << record.threads << ','
        << record.iterations << ','
        << (record.converged ? 1 : 0) << ','
        << record.time_s << ','
        << record.compute_s << ','
        << record.vector_comm_s << ','
        << record.reduction_s << ','
        << record.relative_residual << ','
        << csv_escape(record.git_commit)
        << '\n';

    out.flush();

    if (!out) {
        throw std::runtime_error(
            "Failed to write timing CSV: " + filename
        );
    }
}
