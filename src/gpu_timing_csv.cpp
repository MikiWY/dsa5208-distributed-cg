#include "gpu_timing_csv.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace {

const std::string header =
    "run_id,n,N,device,preprocessing_s,h2d_s,"
    "time_s,d2h_s,iterations,converged,git_commit";

std::string escape_csv(const std::string& value) {
    if (value.find_first_of(",\"\r\n") ==
        std::string::npos) {
        return value;
    }

    std::string result = "\"";

    for (char c : value) {
        if (c == '"') {
            result += "\"\"";
        } else {
            result += c;
        }
    }

    result += '"';
    return result;
}

} // namespace

void append_gpu_timing_csv(
    const std::string& filename,
    const GpuTimingRecord& record
) {
    if (filename.empty()) {
        throw std::invalid_argument(
            "GPU timing CSV path cannot be empty"
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
                "GPU timing CSV schema mismatch"
            );
        }

        needs_header = first_line.empty();
    }

    std::ofstream out(filename, std::ios::app);

    if (!out) {
        throw std::runtime_error(
            "Cannot open GPU timing CSV"
        );
    }

    if (needs_header) {
        out << header << '\n';
    }

    out << std::setprecision(17)
        << escape_csv(record.run_id) << ','
        << record.n << ','
        << record.N << ','
        << escape_csv(record.device) << ','
        << record.preprocessing_s << ','
        << record.h2d_s << ','
        << record.time_s << ','
        << record.d2h_s << ','
        << record.iterations << ','
        << (record.converged ? 1 : 0) << ','
        << escape_csv(record.git_commit)
        << '\n';

    out.flush();

    if (!out) {
        throw std::runtime_error(
            "Failed to write GPU timing CSV"
        );
    }
}
