#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build}"
MPIRUN="${MPIRUN:-mpirun}"
NP="${NP:-2}"

SERIAL="$BUILD_DIR/cg_serial_benchmark"
MPI_BENCH="$BUILD_DIR/cg_mpi_benchmark"

test -x "$SERIAL"
test -x "$MPI_BENCH"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

export OMP_DYNAMIC=FALSE

echo "===== Serial smoke test ====="

"$SERIAL" \
  --n 32 \
  --run-id serial-smoke \
  --timing-output "$TMP_DIR/serial.csv"

echo "===== V1-V4 smoke tests ====="

for solver in mpi hybrid; do
  for comm in allgatherv halo; do

    echo "Testing $solver $comm"

    if [[ "$solver" == "hybrid" ]]; then
      export OMP_NUM_THREADS=2
    else
      export OMP_NUM_THREADS=1
    fi

    "$MPIRUN" -np "$NP" "$MPI_BENCH" \
      --n 32 \
      --solver "$solver" \
      --comm "$comm" \
      --tol 1e-8 \
      --max-iterations 500 \
      --run-id "${solver}-${comm}-smoke" \
      --output "$TMP_DIR/correctness.csv" \
      --timing-output "$TMP_DIR/timing.csv"

  done
done

echo "===== Validate CSV records ====="

python3 - "$TMP_DIR" <<'PY'
import csv
import math
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True
).strip()

with (root / "timing.csv").open(newline="") as f:
    reader = csv.DictReader(f)
    assert len(reader.fieldnames) == 14
    rows = list(reader)

assert len(rows) == 4

expected = {
    ("mpi", "allgatherv"),
    ("mpi", "halo"),
    ("hybrid", "allgatherv"),
    ("hybrid", "halo"),
}

seen = set()

for row in rows:
    run_id = row["run_id"]

    for solver, comm in expected:
        if run_id == f"{solver}-{comm}-smoke":
            seen.add((solver, comm))

    assert row["git_commit"] == commit
    assert row["converged"] == "1"
    assert int(row["iterations"]) == 74
    assert float(row["relative_residual"]) <= 1e-8

    for key in (
        "time_s",
        "compute_s",
        "vector_comm_s",
        "reduction_s",
    ):
        assert row[key] != "", (run_id, key)
        assert math.isfinite(float(row[key]))
        assert float(row[key]) >= 0.0

assert seen == expected

with (root / "correctness.csv").open(newline="") as f:
    reader = csv.DictReader(f)
    assert len(reader.fieldnames) == 15
    correctness = list(reader)

assert len(correctness) == 4

for row in correctness:
    assert row["git_commit"] == commit
    assert row["pass"] == "1"
    assert row["converged"] == "1"
    assert float(row["true_rel_residual"]) <= 1e-8
    assert float(row["relative_solution_error"]) <= 1e-6

with (root / "serial.csv").open(newline="") as f:
    serial = list(csv.DictReader(f))

assert len(serial) == 1
assert serial[0]["git_commit"] == commit
assert serial[0]["converged"] == "1"

print("PASS: Four CPU configurations")
print("PASS: 15-column correctness CSV")
print("PASS: 14-column timing CSV")
print("PASS: V1-V4 component timing")
print("PASS: Embedded Git commit")
print("PASS: Serial benchmark")
PY

echo "PASS: CPU release smoke test"
