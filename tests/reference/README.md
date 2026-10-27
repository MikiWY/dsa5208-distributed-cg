# DSA5208 Independent Poisson Matrix Validation — corrected v2

**Correction:** The previously distributed `poisson_n4_spmv.csv` had six values that were 1 too high. The dense matrix was correct. This version regenerates both n=3 and n=4 reference files and checks dense matrix-vector multiplication against an independent direct five-point stencil using exact integer arithmetic.

## Specification
- Square n x n interior grid; N=n*n; h=1/(n+1).
- Zero-based row-major index k=j*n+i.
- Diagonal = 4/h^2; existing four-neighbor coefficients = -1/h^2; homogeneous Dirichlet boundary.
- Test vector p[k]=k+1.
- CSV matrix has no header. SpMV CSV columns: global_index,p,A_times_p.
- CSR JSON uses zero-based global column indices, sorted ascending within each row.

## Exact SpMV expected output
- n=3: [-32,-16,64,48,0,112,256,176,352]
- n=4: [-75,-50,-25,125,100,0,0,225,200,0,0,325,725,450,475,925]

## Validation procedure
1. Run `python build.py` to regenerate all references (requires NumPy).
2. Compare C++ CSR dense-equivalent entries with matrix CSV; check CSR invariants.
3. Compare serial C++ SpMV to the SpMV CSV using p[k]=k+1.
4. Compare MPI Allgatherv and Halo Exchange SpMV to the same reference, including uneven partitions where feasible.
5. Check ownership ranges cover each global row exactly once.

This dataset validates matrix construction and SpMV only; it does not by itself validate CG convergence.
