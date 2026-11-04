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

Fix A, local Cython branch `prange-fix-end-barrier` (8791d5c54, on 7b3aab363): emit
`nowait` only when `acquire_gil` or for a standalone `prange`. GCC already drops the
end barrier of a standalone loop, clang doesn't, so keeping `nowait` there avoids
adding one on clang. New test `test_prange_end_barrier` in
`tests/run/parallel.pyx`: `Got: 3` without the patch, passes with it. `run.parallel`,
`run.parallel_with_gil`, `run.sequential_parallel`, `run.parallel_swap`,
`run.pure_parallel`, `errors.e_cython_parallel` pass in C mode with GCC 15.
A first `runtests.py` run selected no tests (`--no-file` excludes them) and printed
"ALL DONE": check the "Ran N tests" line.

Draft issue on the fork, not upstream. I wrote references to upstream as plain `GH-N` to leave
no cross-reference on the upstream threads, and that was wrong: GitHub links them from a fork too.
The upstream timelines of GH-6562 and GH-7178 show cakedev0/cython#1, #2, #3 and #4.

## Exp 5: Fix B, and a hang on clang (2026-11-04)

Fix B, local Cython branch `prange-no-start-barrier` (492a49592, on Fix A): `private(x)`
instead of `firstprivate(x) lastprivate(x)` when the variable is a local number or pointer,
not in a closure, without `&x`, all the reads in the body only see assignments of the
body, and no read outside of the body sees one of them. Decided from `cf_state` of each
reference, so two loops reusing `i` don't block each other.

`tests/run/parallel_private_clauses.pyx` covers both directions with
`test_assert_c_code_has`: unused target, reused target, overwritten target, body-only local,
nested target become `private`. Target or local read after, conditional assignment, read
before write, `&x`, `else` clause, `break`, read in the `parallel` block, read in the next
iteration of an outer loop, `locals()`, nested read, closure, module-level target keep the
clauses. GCC 15 through `runtests.py`, all pass: in C parallel, parallel_with_gil,
sequential_parallel, parallel_swap, pure_parallel, e_cython_parallel,
parallel_private_clauses. In C++ (`--no-c`) parallel, parallel_with_gil, sequential_parallel,
parallel_swap, parallel_private_clauses, cpp_locals_parallel.

`runtests.py` doesn't enable OpenMP on clang ("not gcc - FIXME"), its runs report
`skipped`. So `scripts/doctests_clang.sh` builds the test modules by hand. clang 21, Fix B:
parallel_with_gil 18/18, parallel_private_clauses 40/40, parallel 114/116. The 2 failures
are `test_pure_mode` needing `import Cython`, same on master.

parallel_with_gil **hangs on master with clang**. `probes/gil_hang.pyx`,
`scripts/gil_hang.sh`, `results/local/gil_hang.txt`, Python 3.13 GIL build, prange holding
the GIL, 20 s timeout:

| Cython | GCC 15, target read after / unused | clang 21, target read after / unused |
| --- | --- | --- |
| 3.3.1 | ok / ok | hang / hang |
| Fix A | ok / ok | hang / hang |
| Fix B | ok / ok | hang / ok |

Cause confirmed by `scripts/gil_hang_trace.sh` (`results/local/gil_hang_backtrace.txt`, gdb
stopped by a `SIGALRM` once stuck): the main thread is in `__kmpc_barrier`, the 3 others in
`take_gil` from the `PyEval_RestoreThread` of the loop body. Its return address is offset
0x77a1 in the extension, right after the `__kmpc_barrier` call that follows
`__kmpc_for_static_fini`, just before `PyEval_SaveThread`. So it's clang's end barrier from
`lastprivate`, with the GIL held. `results/local/gil_hang_matrix.txt`, 5 runs each: hangs on
clang 21 and 14, Python 3.13 and 3.14, Cython 3.3.1 and master. Returns 1000 on free-threaded
3.13 with clang 21, and on GCC 15. Separate from the race.

## Exp 6: A/B of scikit-learn builds, first round (2026-11-04)

`scripts/host_all.sh`. Builds: `m331`, `mA`, `mB` are scikit-learn main d5a61d1149cb with
Cython 3.3.1, Fix A, Fix A+B. `f331`, `fB` are `hgb/fewer-regions` 9ade78bf1a25 with 3.3.1
and Fix A+B. `nb331` is `hgb/fewer-regions-nb` 10b5afac1433 (manual chunks) with 3.3.1.
`other_ab.py` runs KMeans, ArgKmin, RadiusNeighbors, t-SNE, HGB predict and records a
`check` value per run, `bench_hgb.py` is the HGB fit A/B of `hgb-regions-experiments`.
3 reps, `results/{laptop,gamer}/`, `scripts/analyze.py`. The `GOMP_barrier-sites` column of
the build logs is wrong (it looks for relocations in a linked `.so`). With `call.*GOMP_barrier`,
`histogram.so` has 4 call sites with 3.3.1, 0 with Fix B on main.

Check values agree across builds (spread below 5e-16). Median fit in s on gamer, msd:

| | f331:pr | fB:pr | f331:pr+fuse3 | fB:pr+fuse3 | nb331:pr+fuse3 |
| --- | --- | --- | --- | --- | --- |
| libgomp | 19.64 | 15.83 | 7.71 | 7.65 | 5.97 |
| libomp | 17.15 | 12.81 | 7.30 | 6.94 | 5.43 |

So unmerged regions gain, merged ones don't reach the manual chunks. `fuse3` runs 3 barriers
per split with 3.3.1 (start of each loop) and still ~3 with Fix B (end of each loop): the
second loop of `split_indices` keeps a call to `GOMP_barrier` although the region ends right
after it, and I think so does the feature loop (GCC merges the end of the three loops of that
function into 1 call site). The manual chunks have 1.

Other cases, 3 reps: `hgb_predict` 0.21 -> 0.15 s on the laptop, 0.073 -> 0.056 s on gamer.
KMeans lloyd on the laptop 0.305 (3.3.1), 0.327 (Fix A), 0.340 s (Fix B). t-SNE on gamer
0.77, 0.87, 0.78 s. 168 of the 169 pragmas still using `firstprivate` were `double_pair`
structs of `_loss`, excluded by the first version of Fix B.

## Exp 7: second round (2026-11-04)

Results in `results/{gamer,laptop,xeon-s1}/*2.jsonl`, summary in `FINDINGS.md`. The last-statement
rule closes the gap with the manual chunks. Round 1 on the laptop didn't hold: HGB predict 27%
faster became 4%, KMeans lloyd 11% slower became 3% faster.

Afterwards the last-statement rule moved into Fix A and the branches were rewritten, then opened
as cakedev0/cython#3 (Fix A) and #4 (Fix B). The commit hashes given above for the fixes are the
old ones. The compiler code of Fix B is identical to the benchmarked `fixB2` (empty
`git diff -- Cython`). The `mA` timings are from the Fix A before the move, which only differs for
a `prange` ending a `parallel` block. Fix A alone hangs on `parallel_with_gil` with clang, as
master does (cakedev0/cython#2).

Audit of scikit-learn main for #1: `scripts/audit_parallel_blocks.py`, summary in `FINDINGS.md`.

TSan on intel-laptop, `scripts/tsan/`, results in `results/tsan/`. Mistakes on the way: a first run
on gamer had 5 failed builds (Tempita imports Cython, missing from the env), and its "0 reports"
meant nothing. The first control (`fctl`, `fuse3` with `nowait` everywhere) crashed with
`DEADLYSIGNAL` and hung until the 1500 s timeout, so it reported nothing. `probe.sh` is the control
that reports, run on thinkpad since the other machines were busy. The check in `checks.sh` for
`_argkmin` matched 2 files; `nm -D` on `_argkmin` alone shows 15 `__tsan_` symbols, so the C++
extensions are instrumented.

Sources for the passive default: `libgomp/config/linux/x86/spincount.h` sets the spin count to 1
if CPUID reports a hybrid CPU, unless `OMP_WAIT_POLICY` or `GOMP_SPINCOUNT` is set. `kmp.h`:
`KMP_DEFAULT_BLOCKTIME (__kmp_is_hybrid_cpu() ? (0) : (200000))`. `env -i` on gamer still reports
`GOMP_SPINCOUNT = '1'`.

Fix B changed (`prange-no-start-barrier`), tree `fixB2`: structs and unions are allowed in plain
`private`, and the last statement of a `parallel` block keeps `nowait`, since only the
end of the region follows it. A struct written field by field never becomes private: Cython
doesn't treat that as an assignment, so it stays shared with no clause, as before.
`scripts/cython_tests.sh`: passes in C and C++ with GCC 15, and by hand with clang 21 (same 2
`test_pure_mode` failures).

Builds: `m324` (the Cython that scikit-learn's pin allows), `m331`, `mA`, `mB2`, `f331`, `fB2`,
`nb331`. New names, so that Meson can't reuse C generated by the previous tree. 6 reps for the
non-HGB cases, 3 for HGB. `scripts/sklearn_tests.sh` runs scikit-learn's tests of the modules
using `prange` on `mB2`, `fB2` with `fuse3`, and `m331`.

## Exp 8: HGB fits on main, by Cython version (2026-11-04)

The HGB session measured 3.3 faster than 3.2.5 on main (its Exp 20) and asked if Fix A keeps that.
My earlier HGB fits only used `f*` and `nb331`, so PR #4's numbers have no fit on main.
`scripts/main_fit_ab.sh`, `main_fit_gamer.sh`, `main_fit_libomp.sh`, `main_fit_check.py`, results in
`results/main-fit/`. Xeon held by other runs, not measured.

Generated C on gamer: `histogram.c` and `splitting.c` of `m331` and `mA` have 0 differing function
bodies, after normalizing blank lines and temporaries. The raw diff is mostly the version string,
since the fix branches are on master (3.4.0a0). Between `m324` and `m331`, the parallel functions
differ by `nowait`, `num_threads(n != 0 ? n : omp_get_max_threads())` and the setup of Python calls.

`scripts/barrier_sites.sh`, `call.*GOMP_barrier` on gamer: 483 total on `m324`, 399 on `m331`, 465 on
`mA`, 67 on `mB2`. HGB modules are the same on the first three. `_loss` is 368, 328, 368, 40, and its
only `parallel` blocks are in `CyHalfMultinomialLoss`, so that doesn't touch `year_prediction_msd`.
`mA` matches `m324` there because code follows these loops.

Result: Fix A keeps 3.3's gain. Fix B on gamer with libomp is slower on main, in 2 rounds, while it
gains on the laptop with the same `llvm-openmp` 23.1.2. Tables in `FINDINGS.md`. The cause of 3.3's
gain over 3.2.x is still unknown: all three phases gain, with no change in the barrier sites of the
HGB modules.

## Exp 9: why Fix B loses on gamer with libomp on main (2026-11-04)

All on main with `year_prediction_msd`, `m331` vs `mB2`, interleaved, median of 5 fits in s unless
stated. Results in `results/main-fit/`. The sysfs snapshots are in `snapshots.tar.gz` in each
directory: unpack them there before running `idle_summary.py`.

**Which configurations lose.** `scripts/gamer_probe.sh`:

| gamer, libomp | 3.3.1 | Fix B |
| --- | --- | --- |
| default | 5.25 | 6.82 |
| default, again | 5.54 | 7.18 |
| `KMP_BLOCKTIME=200` (spins) | 2.94 | 2.54 |
| `OMP_PROC_BIND=true OMP_PLACES=cores` | 26.57 | 23.36 |
| 8 threads on P-cores | 4.51 | 4.31 |
| 16 threads on E-cores | 4.34 | 3.43 |

Binding is ~5x slower on both builds, and I don't know why.

**Xeon, one socket, 86 threads.** `scripts/xeon_main_fit.sh`, in the `FINDINGS.md` table. Fix B wins
with libgomp, libomp and `KMP_BLOCKTIME=0`, which is passive like gamer's default. So the slowdown
doesn't come with libomp or passive waiting alone.

**Hypothesis 1, threads stacking on a CPU: refuted.** I thought the barrier gave the scheduler a
second chance to place threads. `probes/barrier_placement.c`, a toy, shows stacking of ~0.04 threads
per region with and without barrier, and the barrier-free version faster. Its first regions were much
longer than HGB's, so `scripts/idle_probe.sh` reruns it at HGB's size (~1 us of work): 27 us per
region without barrier, 31 us with it. So the toy doesn't reproduce the slowdown at either size.

**Where the time goes.** `probes/region_shim.c`, an `LD_PRELOAD` shim around `GOMP_parallel` and
`GOMP_barrier`, on unmodified builds. Totals in s over 87367 regions, gamer, libomp, 3.3.1 / Fix B:

| | wall | wake | spread | work_mean | exit |
| --- | --- | --- | --- | --- | --- |
| mix, 24 threads | 5.62 / 8.34 | 1.66 / 6.09 | 1.24 / 5.67 | 1.09 / 1.20 | .54 / 1.00 |
| E-cores, 16 threads | 4.20 / 3.34 | .61 / .64 | .27 / .33 | 2.03 / 2.04 | .37 / .40 |
| P-cores, 8 threads | 4.81 / 5.53 | .70 / 1.88 | .41 / 1.64 | 2.55 / 2.64 | .33 / .85 |

Wake goes from ~19 us to ~70 us per region, on all the short loops rather than one site, and work
doesn't change. Under the shim, 8 threads on P-cores lose, while without it (above) they don't. I
don't have an explanation for that gap.

**Hypothesis 2, deep idle states.** `scripts/idle_probe.sh`, counters around each fit, base runs:

| P-cores, entries per fit | POLL | C1 | C2 | C3 |
| --- | --- | --- | --- | --- |
| 3.3.1 | 53k | 1642k | 19k | 60k |
| Fix B | 6k | 677k | 41k | 141k |

So each thread sleeps about half as often, since it no longer sleeps in the barrier, and more of
those sleeps go deep (4.5% to 21% of entries in C2 or C3). E-cores show the same rise in C3. This is
only a correlation. The mean stay in C3 is ~290 us, far below its 3144 us target residency, so the
governor is often wrong. The reported exit latency of 1048 us also doesn't match a mean wake of ~60
us per region with ~4 C3 entries per region. The snapshots cover process start and data loading too,
so I can't read that per region strictly.

**SCHED_IDLE spinners: failed as a test.** Meant to keep cores out of idle states and yield at once.
Fits go to ~554 s and ~281 s, with 2.3 threads stacked per region and ~1.8 ms per wake. Fix B is 2x
faster, similar to the Xeon with blocktime 0. Idle state counters are 0, as intended, but the
stacking dominates everything else.

**Timer threads.** `probes/keepawake.c`, one `SCHED_IDLE` thread per CPU sleeping on a periodic timer,
`scripts/keepawake_probe.sh`. No root, and the CPU stays free for wake-up placement. 200 us is below
C2's target residency. I meant 5 ms as a control, above C3's target, but it isn't neutral:

| timer | C3 entries, 3.3.1 / Fix B | fit | fit with shim | wake with shim | work_mean with shim |
| --- | --- | --- | --- | --- | --- |
| none | 131k / 359k | 5.15 / 7.09 | 5.93 / 7.94 | 1.61 / 5.25 | 1.09 / 1.18 |
| 5 ms | 5k / 5k | 4.04 / 3.16 | 4.41 / 3.41 | .92 / 1.25 | 1.06 / 1.07 |
| 200 us | 0 / 0 | 4.01 / 2.79 | 4.38 / 3.20 | .78 / .90 | 1.13 / 1.10 |

Fix B goes from 38% slower to 30% faster, and work doesn't change, so this isn't compute speed while
running. It also fits the toy: it enters C3 ~0.9 times per region without barrier, the fit ~4 times.

Why a 5 ms timer removes nearly all C3 entries isn't clear. My guess is that `menu` scales its
prediction per bucket of next timer distance, so short real sleeps then teach it a short prediction.
I haven't checked that.

**Laptop.** Its idle table (Ultra X7 358H, same driver, governor and C1 to C3 latencies) is identical
to gamer's, yet Fix B wins there, so it could have refuted the hypothesis.
`scripts/keepawake_laptop.sh`, 16 threads, libomp, P-cores are 0 to 3:

| timer | C3 entries, 3.3.1 / Fix B | C1 entries, 3.3.1 / Fix B | fit |
| --- | --- | --- | --- |
| none | 60k / 62k | 3491k / 1705k | 15.27 / 10.47 |
| 5 ms | 5k / 5k | 3572k / 1771k | 15.46 / 10.86 |
| 200 us | 0 / 0 | 4395k / 2508k | 16.81 / 12.14 |

Fix B halves the sleeps here too, but C3 doesn't rise, and removing C3 gains nothing. That's what the
hypothesis predicts, since gamer reaches 359k C3 entries with Fix B. I don't know why the governor goes
deep on one machine and not the other with the same table. The fit is ~3x slower on the laptop, so
the mix of sleep lengths may differ. The 200 us timer costs ~10% here, while it didn't cost anything
visible on gamer.

**Longer timer periods.** `scripts/keepawake_dose.sh`, gamer, looking for a helper that leaves deep
states on. The ssh connection dropped but the run finished on gamer, so there's no `run.log`, and the
fit times are in the `jsonl` files. P-cores and E-cores summed, models identical across builds:

| timer | C3 entries, 3.3.1 / Fix B | C2 entries, 3.3.1 / Fix B | fit | Fix B vs 3.3.1 |
| --- | --- | --- | --- | --- |
| none | 161k / 367k | 68k / 137k | 5.59 / 6.94 | +24% |
| 200 ms | 93k / 153k | 42k / 69k | 4.81 / 4.63 | -4% |
| 50 ms | 34k / 39k | 34k / 40k | 4.30 / 3.40 | -21% |
| 20 ms | 6k / 6k | 28k / 30k | 4.05 / 3.04 | -25% |
| 5 ms | 5k / 5k | 17k / 23k | 4.03 / 3.07 | -24% |

No period is fully neutral: even 200 ms halves C3 on Fix B. But the penalty returns gradually as C3
returns, and 5 wake-ups per second on each CPU can't do much directly, so I read this as an effect on
idle states. A least squares line over these 5 points gives 3.3.1 = 4.0 s + 9.8 us per C3 entry and
Fix B = 3.0 s + 10.8 us per C3 entry. So Fix B saves ~1 s and loses ~2 s to ~200k more C3 entries.
The zero-C3 estimates match the 200 us arm (4.01 s and 2.79 s).

The slope isn't a measured exit latency: a region's wake is the maximum over its threads, entries of
the same region overlap, and the snapshots include process start and data loading. C2 moves with C3,
so I can't separate them.

**Conclusion.** Deep idle states explain Fix B's slowdown on gamer with libomp: well supported, without
a direct test. Disabling states through sysfs or holding `/dev/cpu_dma_latency` would be one, and both
need root. Still unknown: why `menu` goes deep with Fix B on gamer and not on the laptop with the same
table, why 8 threads on P-cores lose only under the shim, and whether `kddcup09_churn` has the same
cause. The Xeon has a different table (C6 at 170 us) and I didn't read its counters.
