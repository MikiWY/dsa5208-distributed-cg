"""Independent Poisson reference generator with exact integer-stencil checks."""
from pathlib import Path
import csv, json
import numpy as np
ROOT = Path(__file__).resolve().parent
for n in (3, 4):
    N = n*n
    scale = (n+1)**2
    A_int = np.zeros((N, N), dtype=np.int64)
    for j in range(n):
        for i in range(n):
            k = j*n+i
            A_int[k,k] = 4*scale
            for ii,jj in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if 0 <= ii < n and 0 <= jj < n:
                    A_int[k,jj*n+ii] = -scale
    p = np.arange(1,N+1,dtype=np.int64)
    q_matrix = A_int @ p
    q_stencil = np.array([
        scale*(4*(j*n+i+1) - sum(jj*n+ii+1 for ii,jj in
        ((i-1,j),(i+1,j),(i,j-1),(i,j+1)) if 0<=ii<n and 0<=jj<n))
        for j in range(n) for i in range(n)],dtype=np.int64)
    assert np.array_equal(q_matrix,q_stencil)
    assert np.array_equal(A_int,A_int.T)
    assert np.linalg.eigvalsh(A_int.astype(float))[0] > 0
    ptr=[0];cols=[];vals=[]
    for k in range(N):
        ids=np.flatnonzero(A_int[k]);cols.extend(ids.tolist());vals.extend(A_int[k,ids].astype(float).tolist());ptr.append(len(cols))
    np.savetxt(ROOT/f'poisson_n{n}_matrix.csv',A_int,delimiter=',',fmt='%d')
    with (ROOT/f'poisson_n{n}_spmv.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['global_index','p','A_times_p']);w.writerows((k,int(p[k]),int(q_stencil[k])) for k in range(N))
    with (ROOT/f'poisson_n{n}_csr.json').open('w') as f:
        json.dump(dict(n=n,N=N,h=1/(n+1),row_ptr=ptr,col_idx=cols,values=vals),f,indent=2)
    print(f'n={n}, q={q_stencil.tolist()}, nnz={len(cols)}')
