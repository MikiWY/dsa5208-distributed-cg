# CPU CG Benchmark Timing Methodology

## Supported configurations

- Serial CG
- V1: MPI-only + MPI_Allgatherv
- V2: MPI-only + Blocking Halo Exchange
- V3: Hybrid MPI+OpenMP + MPI_Allgatherv
- V4: Hybrid MPI+OpenMP + Blocking Halo Exchange

## Total solver time

For distributed configurations, CG-only time is measured
using MPI_Wtime().

All ranks synchronize before the timed solver call.

The reported time_s is the maximum elapsed solver time
across participating MPI ranks:

    time_s = max_r(T_solver,r)

The timed solver call includes initialization, input
validation, CG iterations, vector communication,
global reductions, and internal runtime overhead.

The timed region excludes:

- Matrix construction
- Manufactured solution and RHS generation
- Halo Plan construction
- Halo CSR vector-position preprocessing
- Independent true-residual validation
- Timing CSV output

Serial CG uses std::chrono::steady_clock.

## Component timing

Component timing is accumulated independently on each
MPI rank during the solver call.

The benchmark reports the maximum accumulated time
across MPI ranks for each component.

### compute_s

Includes instrumented local operations:

- Local CSR sparse matrix-vector multiplication
- Local dot-product computation
- CG solution and residual vector updates
- CG search-direction updates

Hybrid configurations include the elapsed wall time
of the corresponding OpenMP operations.

Other initialization, scalar arithmetic, allocation,
and validation overhead may remain unclassified.

### vector_comm_s

For Allgatherv configurations:

- Measures the MPI_Allgatherv call.

For Halo configurations:

- Measures the complete blocking Halo value-exchange call.
- Includes communication-related input validation,
  buffer preparation, MPI_Sendrecv, and value handling.

These boundaries differ and should not be interpreted
as identical measurements of pure network transfer time.

### reduction_s

Includes instrumented MPI_Allreduce operations used
for CG global dot products.

For MPI-only dot products, this also includes the
collective vector-size validation reduction.

Some other solver-level input-validation collectives
are not individually classified.

## Aggregation

Each component is accumulated per rank, then reduced
independently using MPI_MAX.

Therefore:

    compute_s + vector_comm_s + reduction_s

is not required to equal time_s.

The component maxima may originate from different ranks.

## CSV schema

The benchmark retains the 14-column CSV schema:

run_id,dim,n,comm,ranks,threads,iterations,converged,
time_s,compute_s,vector_comm_s,reduction_s,
relative_residual,git_commit

Serial benchmark component timing fields remain empty
until separate serial instrumentation is implemented.

## Numerical validation

The current manufactured-solution tests require:

- True relative residual <= 1e-8
- Relative solution error <= 1e-6

Correctness validation is performed outside the
CG-only timed region.

## Benchmark interpretation

Small problems may be dominated by MPI latency,
OpenMP runtime overhead, memory allocation, and
timing instrumentation.

Formal scaling analysis should use repeated runs,
consistent hardware placement, fixed compiler flags,
and documented MPI/OpenMP configurations.

Single-run local measurements are for functional
validation only.

## Versioning

Git commit metadata is embedded at compile time
using DSA5208_GIT_COMMIT.

Release binaries must be rebuilt after the final
source commit so the recorded commit matches
the source used for compilation.
