# GEMM experiments

Tracking branch for two questions (not meant to be merged into `main`):

1. MKL is sometimes faster than OpenBLAS and sometimes slower: where, why, and
   what change on each side would get the best of both?
2. GEMM throughput stops scaling above some number of cores: is that intrinsic
   (memory bandwidth, problem size) or fixable in the library?

Machines: `intel-xeon` socket slot (`s0`/`s1`, 86 cores / 172 threads,
Granite Rapids, AVX-512 + AMX), `intel-gamer-gpu` (Core Ultra 9 285, 8 P +
16 E cores, AVX2), `mac-m2` (4 P + 4 E cores, Accelerate / AMX).

## Layout

- `pixi.toml`: standalone workspace with every BLAS installed side by side
  (conda-forge OpenBLAS pthreads and OpenMP builds, MKL, BLIS).
- `src/gemmbench.c`: loads a BLAS with `dlopen` and times `cblas_?gemm` on a
  list of shapes (column-major `MxNxK:TT`, see the header comment for the
  mapping from numpy's row-major `X @ Y`).
- `scripts/sweep.py`: runs variants x thread counts x shapes, interleaving the
  variants within each round; writes `results/<machine>/<name>.csv` and a
  `.meta.json` with the command, CPU info and library versions.
- `LOG.md` (and `LOG-mac.md`): chronological experiment log, one entry per
  run with its purpose, command and outcome.
- `FINDINGS.md` (and `FINDINGS-mac.md`): the conclusions so far.

## Running

    pixi install -a && pixi run build
    benchlock --tag "gemm ..." s1 -- pixi run python scripts/sweep.py ...
