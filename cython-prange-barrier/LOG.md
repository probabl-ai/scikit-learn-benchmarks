# Experiment log

Cython sources are `git archive` exports of the tags, run from source with
`PYTHONPATH`. Compilers come from conda-forge through `pixi exec`. Upstream
master was 7b3aab363 (2026-11-03).

## Step 1: history and upstream search (2026-11-04)

No issue or PR reports the barrier as a cost. Searched cython/cython for
`prange barrier`, `firstprivate`, `lastprivate`, `omp barrier`, `nowait`,
`prange deadlock`, `prange uninitialized`, `prange slow`, plus every issue
matching `prange` since 2026-06-01. The first `gh search issues` pass returned
empty for queries that the REST search API matches, so all of these come from
`gh api search/issues`.

On scikit-learn, #12807 and #14392 suspected in 2018-2019 that Cython's extra
OpenMP clauses hurt scaling. Nobody identified the barrier there. #24076 and
#20254 only show barrier symbols in profiles.

History of the clauses, from `git log -S`/`-G` on `Cython/Compiler/Nodes.py`:

| commit | date | release | change |
| --- | --- | --- | --- |
| 508d7cc6e | 2011-05-04 | 0.15 era | `firstprivate(target)` added, "shut up as much compiler warnings as possible". Other variables stay `lastprivate` only |
| 8d9bc082f (GH-7178) | 2025-10-16 | 3.2 | `firstprivate` + `lastprivate` on all non-Python variables, on purpose: "variables that are assigned within the prange can also be pre-initialized". Invalid-value initialization and `save_parallel_vars` removed |
| d68d1721f (GH-6562) | 2026-06-23 | 3.3 | `omp for` becomes `omp for nowait` for every prange. The replacement `omp barrier` is emitted only when `acquire_gil` |

So since GH-7178 `firstprivate` has a documented meaning
(CHANGES: "Variables assigned inside of prange loops can now be initialised
outside of the loop"), tested by `test_init_prange`. It is only a warning
workaround for the target.

`save_lastprivates_label` is dead code since GH-7178: defined, never used.

GCC and clang source (master, fetched 2026-11-04) and the spec agree with the
"copy in before write back" explanation:

- `gcc/omp-low.cc`, `lower_rec_input_clauses`: "for variables in both
  firstprivate and lastprivate clauses we need to ensure the lastprivate
  copying happens after firstprivate copying in all threads".
- `clang/lib/CodeGen/CGStmtOpenMP.cpp`: `EmitOMPFirstprivateClause` returns
  `FirstprivateIsLastprivate && ...`, then "Emit implicit barrier to
  synchronize threads and avoid data races on initialization of firstprivate
  variables and post-update of lastprivate variables".
- OpenMP 5.2, 5.4.5 lastprivate: "If a list item appears in both firstprivate
  and lastprivate clauses, the update required for the lastprivate clause
  occurs after all initializations for the firstprivate clause."

Clang has a second barrier. `EmitOMPForDirective`: "Emit an implicit barrier
at the end. `if (!nowait || HasLastprivates)`". Clang ignores `nowait` on a
loop with `lastprivate`.

## Exp 1: barriers per clause combination (07:30)

`scripts/compilers.sh`, `probes/clauses.c`, `-O2 -fopenmp -c`, counts in the
outlined function. `results/local/compilers.txt`. MSVC through Compiler
Explorer, `scripts/msvc_godbolt.py`, `results/msvc_godbolt.txt`.

| compiler | firstprivate | lastprivate | both | no clause |
| --- | --- | --- | --- | --- |
| GCC 7.5, 8.5, 9.5, 11.4, 13.3, 13.4, 14.4, 15.3 | 0 | 0 | 1 (start) | 0 |
| clang 11.1, 14.0, 18.1, 20.1, 21.1 | 0 | 1 (end) | 2 (start, end) | 0 |
| MSVC v19.latest x64, `/O2 /openmp` | 0 | 0 | 0 | 0 |

The MSVC control without `nowait` shows `_vcomp_barrier` in all four, so the
detection works. MSVC was compile-only: nothing was run on Windows.

"Any firstprivate creates a barrier" in the Nodes.py comment is wrong on all
three.

## Exp 2: does the warning still exist?

`probes/control.c`, `scripts/control.sh`, `results/local/control.txt`. GCC 7.5
to 15.3 warn on a `private` read before write, and on a plain C loop, but not
on `lastprivate`, even on a real read before write. GCC has suppressed it on
purpose since 2187f2a261 (2016-08-08), "PR middle-end/72781 (-Wuninitialized
false positives in OpenMP code)": "Set TREE_NO_WARNING on the private vars for
lastprivate". GCC before 7 is not tested (conda-forge stops at 7.5). Clang 21
gives no warning on the control.

`probes/warn.pyx` through Cython 3.3.1, with sed variants of the generated C:
`warn_nofirst` (drop `firstprivate`) and `warn_private` (replace both by
`private`). `scripts/warn_matrix.sh`, `results/local/warn_matrix.txt`,
`-O2`/`-O3 -Wall -Wextra`, GCC 8, 11, 13, 15, clang 14, 21. The clang 14 rows
of that file failed on `omp.h`, rerun by hand with `-I$CONDA_PREFIX/include`
(script fixed since): same warning as clang 21.

| variant | uninitialized warnings on GCC | on clang |
| --- | --- | --- |
| as generated | `tmp` (`with_parallel_tmp`) | `i` (`with_break`) |
| `warn_nofirst` | same, no new one | same, no new one |
| `warn_private` | same plus `i` read after the loop in `with_break`, `x` in `read_before_write` | same, no new one |

Those two new GCC warnings are correct: they are the cases where the clauses
have to stay. Note the `tmp` one on `cdef double tmp; with parallel: tmp = 1`,
a side effect of GH-7178 making `parallel` privates `firstprivate`. Unrelated
to this work.

Barrier calls on the same C (`results/local/barriers_generated.txt`): as
generated, GCC 15 has 1 per prange (3 call sites in the function with two
pranges, not understood yet) and clang 21 has 2. `warn_nofirst`: GCC 0, clang 1.
`warn_private`: 0 on both.

## Exp 3: race after a prange in a `with parallel` block

`probes/race.pyx`, `scripts/race.sh`, `results/local/race.txt`. 4 threads,
2000 items, the thread owning the last chunk sleeps 20 ms before its first
item, 10 calls. First version of the probe slept on every item of that chunk
and was killed after ~8 minutes.

| Cython | compiler | threads seeing unwritten items right after the loop | items unwritten at the second prange |
| --- | --- | --- | --- |
| 3.1.2, 3.2.4 | GCC 15 | 0 in all 10 | 0 |
| 3.1.2, 3.2.4 | clang 21 | 0 in all 10 | 0 |
| 3.3.1 | GCC 15 | **3 in all 10** | 0 |
| 3.3.1 | clang 21 | 0 in all 10 | 0 |

This is a regression of Cython 3.3 with GCC, independent of the barrier cost:
GH-6562 removed the end barrier of the worksharing loop in nogil code. Between
two pranges, only the start barrier of the second one synchronizes. MSVC has no
barrier there at all (Exp 1), so I expect the second column to fail there too,
not tested. master has the same code.

Census of scikit-learn (6e9d6678e7): 62 pranges, 31 standalone, 31 inside a
`with parallel` block. All of HGB's are standalone on main. scikit-learn pins
`cython<3.2.6` in `pyproject.toml`, so its wheels have the end barrier. The
pixi `sklearn-dev` env has Cython 3.3.0. I have not yet audited its 31 loops in
`with parallel` for reads after the loop.

## Exp 4: reproducer for the issue, Fix A (2026-11-04)

`probes/issue_reproducer.pyx`, `scripts/issue_reproducer.sh`,
`results/local/issue_reproducer.txt`. Same outcome as Exp 3, also on master: 3
threads in all 10 calls with GCC 15, 0 with clang 21.

Fix A, `cakedev0/cython@prange-fix-end-barrier` (local for now): emit `nowait`
only when `acquire_gil`. New test `test_prange_end_barrier` in
`tests/run/parallel.pyx`: `Got: 3` without the patch, passes with it. `run.parallel`,
`run.parallel_with_gil`, `run.sequential_parallel`, `run.parallel_swap`,
`run.pure_parallel`, `errors.e_cython_parallel` pass in C mode with GCC 15.
A first `runtests.py` run selected no tests (`--no-file` excludes them) and printed
"ALL DONE": check the "Ran N tests" line.

Draft issue on the fork, not upstream. References to upstream are plain `GH-N`
text on purpose, to leave no cross-reference on the upstream threads.
