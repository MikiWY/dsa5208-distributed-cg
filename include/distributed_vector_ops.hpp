#pragma once

#include <mpi.h>
#include <vector>

// Compute the global dot product of distributed vectors.
//
// Each rank provides only its locally owned vector entries.
// Local vector lengths may differ across ranks.
//
// MPI_Allreduce(MPI_SUM) combines local dot products.
// The returned scalar is identical on all ranks,
// up to the result of the collective reduction.
//
// Ghost entries must not be included in local dot products.

double distributed_dot_product(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm
);
