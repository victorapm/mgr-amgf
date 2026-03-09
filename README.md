# Minimal AMGF-via-MGR Driver

This directory contains a small MPI driver that solves `Ax=b` with:

- outer solver: hypre `ParCSRGMRES`
- preconditioner: a minimal AMGF-style two-level strategy implemented through hypre MGR
- level-0 G-relaxation: BoomerAMG
- coarsest solver: hypre MGR direct solver wrapper (SuperLU_DIST backend)

The implementation is intentionally minimal and standalone from `hypredrive` source changes.
For the AMGF wrapper code itself, required external libraries are hypre (with MGR) and SuperLU_DIST (through hypre's DSUPERLU path), plus MPI.

## What is in this folder

- `amgf_mgr_minimal.c`, `amgf_mgr_minimal.h`: minimal AMGF-via-MGR preconditioner wrapper
- `mgr_driver_gmres.c`: driver executable; reads matrix/vector from IJ files and runs GMRES
- `build.sh`: builds SuperLU_DIST, hypre, and this driver under `external/`
- `test/blocks-unsym-ls_00001`: sample matrix/rhs/dofmap inputs

## Scope and design choices

This is a focused driver for experimenting with the AMGF idea through existing hypre MGR APIs, not a full reproduction of all AMGF research variants.

- MGR hierarchy depth is fixed to two levels (one elimination level + coarse solve).
- The C/F split is user-provided through a local integer mask file (`dofmap`).
- Transfer and relaxation settings are fixed in code to a minimal robust setup.
- The only runtime knob added for MGR internals is `--mgr-print-level`.

## AMGF-via-MGR mapping in this implementation

Using MGR nomenclature, the preconditioner corresponds to:

- level 0: C-points/F-points are set from `point_marker_array` derived from `dofmap`
- level 0: G-relaxation uses BoomerAMG as a global smoother
- coarsest level: sparse direct solve through `HYPRE_MGRDirectSolver*` (requires hypre built with DSUPERLU)

Current fixed settings in `amgf_mgr_minimal.c` include:

- `HYPRE_MGRSetMaxIter(..., 1)`, `HYPRE_MGRSetTol(..., 0.0)`
- `HYPRE_MGRSetMaxGlobalSmoothIters(..., 1)`
- interpolation/restriction/coarse-grid method arrays set to `0` (injection/GLK-RAP defaults)
- BoomerAMG G-relaxation configured with coarsen type `6`, relax type `8`, max iter `1`, tol `0.0`

## SuperLU_DIST requirement check

The preconditioner has an explicit check for DSUPERLU support:

- compile-time/runtime guard via `HYPRE_USING_DSUPERLU`
- clear error message if hypre was not built with SuperLU_DIST

`build.sh` also configures hypre with:

- `-DHYPRE_ENABLE_DSUPERLU=ON`
- `-DTPL_DSUPERLU_INCLUDE_DIRS=...`
- `-DTPL_DSUPERLU_LIBRARIES=...`

## Build

Run from this directory:

```bash
./build.sh --build-type Release
```

Common options:

- `--build-type <Release|Debug|RelWithDebInfo|...>` (default: `Release`)
- `--jobs <N>`
- `--clean` (removes `external/src`, `external/build`, and `external/install` first)
- `--refresh-sources` (force-fetches requested git refs even if already cached)
- `--hypre-tag <tag>` (default: `v3.1.0`)
- `--superlu-tag <tag>` (default: `v9.2.1`)

Build artifacts are placed under `external/`:

- `external/install/superlu_dist`
- `external/install/hypre`
- `external/build/mgr_driver/mgr_driver`

## Input format

The driver reads:

- matrix with `HYPRE_IJMatrixRead` from `--matrix-prefix`
- rhs with `HYPRE_IJVectorRead` from `--rhs-prefix`
- optional initial guess with `HYPRE_IJVectorRead` from `--x0-prefix`
- C/F marker file from `--dofmap-prefix`

`dofmap` format expected by this driver:

- first value: local row count `n`
- next `n` integer entries: per-row marker
- marker convention: `0` -> F-point, nonzero -> C-point (internally normalized to `1`)

For MPI ranks, it first tries `<prefix>.<rank:05d>`, then falls back to `<prefix>`.

## Run

Default test input:

```bash
mpirun -np 1 ./external/build/mgr_driver/mgr_driver
```

Example with explicit options:

```bash
mpirun -np 1 ./external/build/mgr_driver/mgr_driver \
  --matrix-prefix test/blocks-unsym-ls_00001/IJ.out.A \
  --rhs-prefix test/blocks-unsym-ls_00001/IJ.out.b \
  --dofmap-prefix test/blocks-unsym-ls_00001/dofmap.out \
  --tol 1e-8 \
  --max-iter 200 \
  --k-dim 50 \
  --print-level 1 \
  --mgr-print-level 1 \
  --logging 1
```

Current CLI options:

- `--matrix-prefix <path>`
- `--rhs-prefix <path>`
- `--dofmap-prefix <path>`
- `--x0-prefix <path>`
- `--tol <real>`
- `--max-iter <int>`
- `--k-dim <int>`
- `--print-level <int>` (GMRES output)
- `--mgr-print-level <int>` (MGR output)
- `--logging <0|1>`
- `-h`, `--help`

If `--x0-prefix` is omitted, the initial guess is set to zero.

## Verified baseline run

Command:

```bash
mpirun -np 1 ./external/build/mgr_driver/mgr_driver --print-level 0 --mgr-print-level 0
```

Full output:

```text
============================================================
AMGF-MGR Driver: Matrix Summary
  MPI tasks: 1
  Matrix source: test/blocks-unsym-ls_00001/IJ.out.A
  Global size: 1590 x 1590
  Global nonzeros: 121806
  Rank 0 local row range: [0, 1589] (1590 rows)
  Rank 0 local col range: [0, 1589] (1590 cols)
  Constraints (dofmap==1): rank0 local=462, global=462
============================================================
GMRES converged.
  Iterations: 25
  Final relative residual norm: 8.7734377970191215e-09
```

## Limitations and gaps

- No automatic C/F splitting: the quality of `dofmap` strongly affects convergence.
- MGR and BoomerAMG internal parameters are mostly hard-coded in `amgf_mgr_minimal.c`.
- The driver reports convergence metrics but does not currently write the solution vector to disk.
- Error handling is simple and mostly rank-0 text reporting.

## Reference

For AMGF motivation and theory, see:

- Socratis Petrides, Tucker Hartland, Tzanio Kolev, Chak Shing Lee, Michael Puso, Jerome Solberg, Eric B. Chin, Jingyi Wang, and Cosmin Petra, *AMG with Filtering: An Efficient Preconditioner for Interior Point Methods in Large-Scale Contact Mechanics Optimization*, manuscript.
- arXiv: https://arxiv.org/abs/2505.18576
