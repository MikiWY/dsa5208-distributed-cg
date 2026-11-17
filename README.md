# DSA5208 Project 3

**Distributed Conjugate Gradient Solver: Parallel Implementation and Performance Evaluation**

A C++17 implementation and performance study of the Conjugate Gradient (CG) method for solving sparse symmetric positive definite linear systems arising from the two-dimensional Poisson equation.

The project evaluates serial execution, MPI-only parallelism, hybrid MPI+OpenMP parallelism, and CUDA GPU execution on the NUS Vanda HPC cluster.

## 1. Project Overview

The project investigates how parallelization and communication strategies affect the numerical correctness, runtime, and scalability of Conjugate Gradient solvers.

The implemented configurations are:

| Configuration | Parallelism | Communication |
|---|---|---|
| Serial | Single CPU process | None |
| V1 | MPI-only | MPI_Allgatherv |
| V2 | MPI-only | Blocking Halo Exchange |
| V3 | Hybrid MPI+OpenMP | MPI_Allgatherv |
| V4 | Hybrid MPI+OpenMP | Blocking Halo Exchange |
| GPU | CUDA on NVIDIA GPU | Device-resident CG |

All implementations solve the same mathematical problem using CSR sparse matrices and double-precision arithmetic.

The evaluation includes serial runtime scaling, MPI strong and weak scaling, communication comparisons, hybrid parallelism, GPU performance, and independent numerical correctness checks.

## 2. Mathematical Specification

The benchmark solves the two-dimensional Poisson equation:

\[
-\Delta u(x,y)=f(x,y), \quad (x,y)\in(0,1)^2
\]

with homogeneous Dirichlet boundary conditions.

A five-point finite-difference discretization produces the linear system:

\[
Ax=b
\]

The matrix is symmetric positive definite (SPD), making it suitable for Conjugate Gradient.

**Problem parameters:**

- Interior grid: `n × n`
- Unknowns: `N = n²`
- Grid spacing: `h = 1/(n+1)`
- Matrix diagonal: `4/h²`
- Neighbor coefficients: `-1/h²`
- Number of nonzeros: `nnz = 5n² - 4n`
- Matrix format: Compressed Sparse Row (CSR)
- Indexing: Zero-based, row-major
- Precision: Double
- Initial guess: `x0 = 0`
- Default CG tolerance: `1e-8`

Manufactured discrete solutions are used for correctness verification. A reference solution `x_true` is constructed and the right-hand side is generated as `b = A*x_true`.

## 3. Implementation

### 3.1 Serial Conjugate Gradient

The serial solver provides a correctness and performance reference.

It implements the standard CG recurrence, including sparse matrix-vector multiplication, dot products, solution and residual updates, convergence checks, and numerical breakdown handling.

### 3.2 MPI Domain Decomposition

The global matrix is distributed using contiguous blocks of global unknown indices.

Each MPI rank owns a subset of CSR matrix rows. These partitions do **not necessarily correspond to complete two-dimensional grid rows**.

The implementation supports uneven partitions and ranks with zero locally owned rows.

Global CG dot products use `MPI_Allreduce`.

### 3.3 MPI_Allgatherv

The Allgatherv strategy reconstructs the distributed search-direction vector on every rank before local SpMV.

This simplifies global CSR indexing but introduces replicated vector storage and communication overhead.

### 3.4 Blocking Halo Exchange

The Halo strategy communicates only the remote vector entries required by locally owned matrix rows.

The implementation uses:

- Ghost-index discovery
- Reusable Halo Exchange plans
- Cached CSR-to-local/ghost vector positions
- Blocking `MPI_Sendrecv`
- Global reductions for CG dot products

The current implementation uses deterministic pairwise rank scheduling. Although Halo Exchange reduces communicated vector volume, the scheduling overhead can become significant at high MPI rank counts.

### 3.5 Hybrid MPI+OpenMP

The hybrid implementations combine distributed-memory MPI with shared-memory OpenMP.

OpenMP parallelizes local SpMV, dot products, and vector updates.

MPI calls are made outside OpenMP parallel regions using the `MPI_THREAD_FUNNELED` model.

### 3.6 CUDA GPU Solver

The CUDA implementation provides a single-GPU CG execution path using:

- Device-resident CSR matrices and CG vectors
- CUDA kernels for SpMV and vector updates
- Thrust reductions for dot products
- A reusable GPU Workspace
- Repeated-solve support
- Separate preprocessing, host-to-device, CG-only, and device-to-host timing measurements

The host controls the CG iteration sequence. GPU reductions may introduce synchronization overhead.

The CUDA implementation is a standalone GPU solver rather than a distributed multi-GPU MPI implementation.

## 4. HPC Environment

The implementation was validated on the **NUS Vanda HPC cluster**.

| Component | Verified environment |
|---|---|
| CPU | Intel Xeon Platinum 8452Y |
| CPU compiler | GCC 13.3.0 |
| MPI | Open MPI 5.0.3 |
| Parallel programming | MPI + OpenMP |
| Scheduler | PBS |
| GPU | NVIDIA A40 |
| CUDA Toolkit | 12.1.105 |
| CUDA host compiler | GCC 12.3.0 |
| CUDA architecture | `sm_86` |

CPU and GPU builds use different host compiler versions because of CUDA toolkit compatibility.

## 5. Build Instructions

All commands below should be executed from the repository root.

### 5.1 CPU Benchmarks

After loading the required GCC and Open MPI environment:

```bash
CXX=g++ MPICXX=mpicxx \
bash scripts/build_cpu_benchmarks.sh
```

This builds:

- `build/cg_serial_benchmark`
- `build/cg_mpi_benchmark`

The MPI executable supports all four V1–V4 configurations through command-line options.

### 5.2 CPU Release Validation

```bash
bash scripts/validate_cpu_release.sh
```

The validation script checks:

- Serial CG
- MPI Allgatherv
- MPI Halo
- Hybrid Allgatherv
- Hybrid Halo
- Numerical convergence
- Correctness and timing CSV schemas
- Populated component timing fields
- Embedded Git commit metadata

For the frozen benchmark release, these checks passed on Vanda.

### 5.3 CUDA GPU Build

On a Vanda GPU compute node, load CUDA 12.1 and the compatible GCC 12.3 compiler.

Confirm the compiler environment before building.

```bash
export NVCC_HOST_CXX="$(command -v g++)"
export CUDA_ARCH=sm_86

bash scripts/build_gpu_tests.sh
```

The build script produces:

- `build/test_gpu_spmv`
- `build/test_gpu_cg`
- `build/test_gpu_workspace`
- `build/gpu_benchmark`

Run the GPU correctness tests:

```bash
./build/test_gpu_spmv
./build/test_gpu_cg
./build/test_gpu_workspace
```

GPU execution requires an allocated NVIDIA GPU node.

## 6. Running Benchmarks

Create an output directory before running benchmarks:

```bash
mkdir -p results
```

### 6.1 Serial CG

```bash
./build/cg_serial_benchmark \
  --n 128 \
  --run-id serial-n128 \
  --timing-output results/serial_timing.csv
```

### 6.2 V1 — MPI Allgatherv

```bash
OMP_NUM_THREADS=1 mpirun -np 4 \
  ./build/cg_mpi_benchmark \
  --n 128 \
  --solver mpi \
  --comm allgatherv \
  --tol 1e-8 \
  --max-iterations 500 \
  --run-id v1-n128 \
  --output results/correctness.csv \
  --timing-output results/timing.csv
```

### 6.3 V2 — MPI Blocking Halo

```bash
OMP_NUM_THREADS=1 mpirun -np 4 \
  ./build/cg_mpi_benchmark \
  --n 128 \
  --solver mpi \
  --comm halo \
  --tol 1e-8 \
  --max-iterations 500 \
  --run-id v2-n128 \
  --output results/correctness.csv \
  --timing-output results/timing.csv
```

### 6.4 V3 — Hybrid Allgatherv

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
mpirun -np 4 ./build/cg_mpi_benchmark \
  --n 128 \
  --solver hybrid \
  --comm allgatherv \
  --tol 1e-8 \
  --max-iterations 500 \
  --run-id v3-n128 \
  --output results/correctness.csv \
  --timing-output results/timing.csv
```

### 6.5 V4 — Hybrid Blocking Halo

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
mpirun -np 4 ./build/cg_mpi_benchmark \
  --n 128 \
  --solver hybrid \
  --comm halo \
  --tol 1e-8 \
  --max-iterations 500 \
  --run-id v4-n128 \
  --output results/correctness.csv \
  --timing-output results/timing.csv
```

### 6.6 CUDA GPU Benchmark

```bash
./build/gpu_benchmark \
  --n 128 \
  --tol 1e-8 \
  --max-iterations 500 \
  --run-id gpu-n128 \
  --output results/gpu_correctness.csv \
  --timing-output results/gpu_timing.csv \
  --gpu-timing-output results/gpu_components.csv
```

For formal performance experiments, MPI ranks, OpenMP threads, CPU affinity, and node placement should be controlled through the HPC job allocation.

The commands above are functional examples, not a substitute for the PBS resource configuration used in the formal experiments.

## 7. Correctness Verification

Numerical correctness is evaluated using the independently recomputed true relative residual:

\[
R_{\mathrm{rel}}=\frac{\|b-Ax\|_2}{\|b\|_2}
\]

and the relative solution error:

\[
E_{\mathrm{rel}}=\frac{\|x-x_{\mathrm{true}}\|_2}{\|x_{\mathrm{true}}\|_2}
\]

The original benchmark acceptance thresholds are:

| Metric | Threshold |
|---|---|
| True relative residual | ≤ `1e-8` |
| Relative solution error | ≤ `1e-6` |

Correctness testing includes manufactured solutions, multiple MPI rank counts, uneven partitions, zero-owned-row cases, zero RHS, iteration limits, and comparisons between solver configurations.

GPU tests additionally check numerical consistency with the CPU implementation and repeated workspace solves.

### Large-grid numerical accuracy

Two large weak-scaling configurations satisfied the residual criterion but exceeded the original solution-error threshold:

| Grid n | MPI ranks | True residual | Solution error |
|---|---:|---:|---:|
| 2048 | 256 | 9.934e-09 | 1.120e-06 |
| 2896 | 512 | 9.858e-09 | 1.873e-06 |

These results remain recorded as failures under the original acceptance policy.

Separate experiments with tolerance `1e-10` were performed to investigate sensitivity to the stopping tolerance. They are not mixed into the original `1e-8` performance series.

## 8. Timing Methodology

The benchmark reports **CG-only solver time**.

For MPI and Hybrid configurations:

1. Synchronize MPI ranks before timing.
2. Measure local elapsed solver time using `MPI_Wtime`.
3. Execute the CG solver.
4. Aggregate elapsed time across ranks using `MPI_MAX`.
5. Perform independent numerical validation outside the timed region.

The CG-only timing excludes matrix generation, RHS construction, Halo Plan preprocessing, independent true-residual validation, and CSV writing.

The distributed benchmark also records:

- `compute_s`
- `vector_comm_s`
- `reduction_s`

Each component is accumulated independently per rank and then aggregated using `MPI_MAX`.

**These component maxima are not additive and must not be interpreted as percentages summing to 100%.**

For GPU, the benchmark additionally records preprocessing and host/device transfer timings. The GPU CG-only runtime excludes initial workspace preparation and final solution download.

Further timing definitions are documented in `docs/timing_methodology.md`.

## 9. CSV Outputs

The benchmark provides three CSV schemas.

### Correctness CSV — 15 columns

```text
run_id,dim,n,N,solver,comm,ranks,threads,iterations,converged,recursive_rel_residual,true_rel_residual,relative_solution_error,pass,git_commit
```

### CPU/General Timing CSV — 14 columns

```text
run_id,dim,n,comm,ranks,threads,iterations,converged,time_s,compute_s,vector_comm_s,reduction_s,relative_residual,git_commit
```

### GPU-specific Timing CSV — 11 columns

The GPU-specific CSV separately records workspace preparation, data transfers, and CG-only timing.

For formal experiments, run IDs and Git commit metadata are used to trace records back to their configurations and source versions.

## 10. Performance Evaluation

The formal experiments were performed on NUS Vanda using repeated measurements.

### 10.1 Serial Scaling

Serial CG was evaluated at:

`n = 64, 128, 256, 512, 1024`

Runtime was analyzed against matrix dimension `N` and theoretical nonzero count `nnz` using log–log plots.

### 10.2 MPI Strong Scaling

For fixed `n = 512`, MPI rank counts were varied from 1 to 512.

Speedup and parallel efficiency are defined as:

\[
S(P)=\frac{T(1)}{T(P)}
\]

\[
E(P)=\frac{S(P)}{P}
\]

The measured results show that both Allgatherv and Halo initially benefit from increasing MPI parallelism, but performance deteriorates at higher rank counts.

### 10.3 MPI Weak Scaling

The weak-scaling experiments targeted approximately 16,384 unknowns per MPI rank.

Since the global problem size increases with the number of ranks, CG iteration counts may also increase. Total runtime should therefore be interpreted alongside iteration counts and mean time per iteration.

The primary passing-only `1e-8` weak-scaling series extends through 128 ranks. Larger configurations were attempted and analyzed separately due to numerical acceptance failures.

### 10.4 Communication Comparison

At `n = 512`, Blocking Halo Exchange achieved its best measured runtime of approximately 0.310 s at 16 MPI ranks.

However, at 512 MPI ranks, the measured Halo runtime increased to approximately 57.698 s, compared with 7.137 s for Allgatherv.

This demonstrates that lower communicated data volume does not necessarily imply lower elapsed communication time. Pairwise communication scheduling and synchronization overhead are possible contributing factors.

### 10.5 Hybrid MPI+OpenMP

MPI-only and Hybrid configurations were compared at fixed total CPU-core counts.

Hybrid execution did not consistently outperform MPI-only execution in the measured configurations. Performance depended on MPI rank count, OpenMP thread count, communication strategy, and runtime overhead.

### 10.6 CUDA GPU Performance

CUDA CG was evaluated on NVIDIA A40 for:

`n = 64, 128, 256, 512`

At `n = 512`:

- Serial CPU mean CG-only time: approximately 1.7185 s
- GPU mean CG-only time: approximately 0.1043 s
- Measured CG-only runtime ratio: approximately **16.48×**

This is a hardware-specific CG-only runtime comparison, not an end-to-end application speedup.

Detailed plots, experimental conditions, and limitations are provided in the final project report.

## 11. Reproducibility

**Frozen benchmark commit:**

```text
409df2e45ef613fc53492fd76af679cdf88bb537
```

CPU and GPU release smoke tests passed on NUS Vanda using this commit.

The formal benchmark version was frozen before performance data collection. Solver algorithms, acceptance thresholds, timing definitions, and CSV schemas were not changed during the experiments.

Reproduction should record the compiler versions, MPI/CUDA environment, grid size, rank/thread counts, CPU/GPU allocation, PBS metadata, and raw benchmark outputs.

## 12. Project Structure

```text
include/    Solver interfaces, CSR structures, timing and CSV definitions
src/        Serial, MPI, Hybrid, Halo and CUDA implementations
tests/      Numerical correctness and regression tests
scripts/    Build and release-validation scripts
docs/       Timing methodology documentation
```

## 13. Limitations and Future Work

The main limitations include:

- Communication and synchronization overhead at high MPI rank counts.
- Pairwise scheduling overhead in the current Blocking Halo implementation.
- Increasing CG iteration counts with grid refinement.
- Differences in CPU and GPU hardware and timing boundaries.
- Variability between some PBS allocations and process placements.
- Large-grid residual convergence without satisfying the original manufactured-solution-error threshold.

Possible future improvements include neighbor-only Halo scheduling, reduced-synchronization CG methods, preconditioning, and more detailed CPU/GPU profiling.

These are prospective improvements rather than features claimed to have been implemented.

## 14. AI Usage

ChatGPT and OpenAI Codex were used to assist with code suggestions, debugging, numerical-result interpretation, and documentation preparation.

The authors reviewed the mathematical formulations, implementation decisions, correctness checks, experimental outputs, and final conclusions, and take responsibility for the submitted work.

## 15. Project Report and Source Code

The final project report contains the mathematical derivations, implementation methodology, correctness validation, performance plots, numerical results, and discussion.

The repository provides the corresponding implementation, test programs, and reproducible build instructions.

For formal academic evaluation, please refer to the submitted project report together with the frozen benchmark commit identified above.