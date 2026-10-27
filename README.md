# DSA5208 Project 3

Performance and Scalability Analysis of a Distributed
Conjugate Gradient Solver Using Hybrid MPI-OpenMP
and Communication Optimization.

## Project Overview

This project studies the parallelization of the
Conjugate Gradient (CG) algorithm for sparse linear systems.

Implementations planned:

- Serial CG
- MPI-only CG
- Hybrid MPI + OpenMP CG

Communication strategies:

- MPI_Allgatherv
- Blocking Halo Exchange

## Mathematical Specification

- Problem: 2D Poisson equation
- Boundary conditions: homogeneous Dirichlet
- Discretization: five-point stencil with 1/h^2
- Grid: n x n interior points
- Total unknowns: N = n^2
- Grid spacing: h = 1/(n+1)
- Precision: double
- Initial guess: x0 = 0
- Default tolerance: 1e-8
- Matrix storage: CSR
- Indexing: zero-based row-major

Correctness testing follows the team's
Correctness Test Plan.

## Parallel Design

- MPI decomposition: contiguous complete grid-row strips
- MPI communication: Allgatherv and Blocking Halo
- OpenMP operations: SpMV, dot products, vector updates
- MPI calls initially outside OpenMP parallel regions

Both communication strategies are planned for
MPI-only and hybrid MPI + OpenMP configurations.

## HPC Environment

Target platform: NUS Vanda

Verified by the HPC team member:

- GCC 13.3.0
- OpenMPI 5.0.3
- PBS scheduler
- MPI + OpenMP test: 2 ranks x 2 threads completed

## Timing Methodology

Solver time includes:

- Initial residual computation
- CG iterations
- MPI communication and reductions
- Vector updates and convergence checks

Solver time excludes:

- Matrix and RHS generation
- Domain partitioning and halo-map construction
- Memory allocation and preprocessing
- Final independent correctness validation
- CSV output

MPI timing:

1. MPI_Barrier before timing.
2. Start local MPI_Wtime timer.
3. Execute CG solver.
4. Stop local timer.
5. MPI_Allreduce with MPI_MAX after timing.

## Validation and Results

The team will use the agreed 15-column CSV schema
from the Correctness Test Plan.

Additional performance measurements will be stored
separately.

All implementation and performance results remain
pending until actual tests are completed.

## Collaboration

Source code is maintained in a shared private
GitHub repository.

Changes to shared interfaces require team review.

## AI Usage

AI-assisted programming and documentation will be
disclosed in the final project report.
All generated code must be compiled, tested, and
reviewed by the team.
