"""Add a split-K path to OpenBLAS interface/gemm.c (prototype).

usage: python apply_splitk_gemm.py OPENBLAS_SRC

When each thread's share of C is small (m * n < nthreads * 128^2) and k is
long (k / nthreads >= 512), split k into one chunk per thread, run the
single-threaded driver on each chunk (chunk 0 writes C with the user's beta,
the others write private buffers with beta = 0), then reduce the buffers into
C in parallel over columns. OPENBLAS_GEMM_SPLITK=0 disables it, =N forces N
chunks (read once).
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "interface" / "gemm.c"
s = path.read_text()
assert "GEMM_SPLITK" not in s, "already patched"

helper = r'''
#if defined(SMP) && !defined(COMPLEX) && !defined(BFLOAT16) && !defined(HFLOAT16) && !defined(GEMM3M) && !defined(XDOUBLE)
#define GEMM_SPLITK 1

struct splitk_job {
  blas_arg_t args;  /* first member: exec_blas passes &job as the args */
  int (*fn)(blas_arg_t *, BLASLONG *, BLASLONG *, IFLOAT *, IFLOAT *, BLASLONG);
  FLOAT *buf;
  BLASLONG nbuf;
};

static int splitk_env = -2;

static BLASLONG splitk_nchunks(BLASLONG m, BLASLONG n, BLASLONG k, int nthreads) {
  if (splitk_env == -2) {
    char *e = getenv("OPENBLAS_GEMM_SPLITK");
    splitk_env = e ? atoi(e) : -1;
  }
  if (splitk_env == 0 || nthreads < 2) return 1;
  BLASLONG p = nthreads;
  if (splitk_env > 0) return MIN(splitk_env, nthreads);
#ifdef SPLITK_V1
  /* The M x N partition already feeds the threads with 32 x 32 tiles. */
  BLASLONG tiles = MAX(1, m / 32) * MAX(1, n / 32);
  if (2 * tiles >= nthreads) return 1;
#else
  /* Each thread's share of C is large enough for the M x N partition. */
  if (m * n >= (BLASLONG)nthreads * 128 * 128) return 1;
#endif
  /* Keep chunks long enough to amortize packing. */
  if (k / p < 512) p = k / 512;
  /* Cap the scratch buffers at 64 MiB. */
  BLASLONG cap = (64L << 20) / (m * n * (BLASLONG)sizeof(FLOAT));
  if (p - 1 > cap) p = cap + 1;
#ifndef SPLITK_V1
  /* With single-threaded chunks, fewer than half the threads would work. */
  if (2 * p < nthreads) return 1;
#endif
  return p < 2 ? 1 : p;
}

static int splitk_worker(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                         IFLOAT *sa, IFLOAT *sb, BLASLONG mypos) {
  struct splitk_job *job = (struct splitk_job *)args;
  return job->fn(&job->args, NULL, NULL, sa, sb, 0);
}

static int splitk_reduce(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                         IFLOAT *sa, IFLOAT *sb, BLASLONG mypos) {
  struct splitk_job *job = (struct splitk_job *)args;
  BLASLONG m = job->args.m, n = job->args.n, ldc = job->args.ldc;
  FLOAT *c = (FLOAT *)job->args.c;
  for (BLASLONG j = range_n[0]; j < range_n[1]; j++) {
    FLOAT *cj = c + j * ldc;
    for (BLASLONG p = 0; p < job->nbuf; p++) {
      FLOAT *bj = job->buf + p * m * n + j * m;
      for (BLASLONG i = 0; i < m; i++) cj[i] += bj[i];
    }
  }
  return 0;
}

static int gemm_splitk(blas_arg_t *args, int transa, int transb, IFLOAT *sa, IFLOAT *sb) {
  BLASLONG m = args->m, n = args->n, k = args->k;
  if (((FLOAT *)args->alpha)[0] == ZERO) return 0;
  BLASLONG p = splitk_nchunks(m, n, k, args->nthreads);
  if (p < 2) return 0;
  FLOAT *buf = (FLOAT *)malloc((size_t)(p - 1) * m * n * sizeof(FLOAT));
  if (buf == NULL) return 0;
  static FLOAT zero = ZERO;
#ifdef DOUBLE
  int mode = BLAS_DOUBLE | BLAS_REAL;
#else
  int mode = BLAS_SINGLE | BLAS_REAL;
#endif
  struct splitk_job jobs[MAX_CPU_NUMBER];
  blas_queue_t queue[MAX_CPU_NUMBER];
  BLASLONG range[MAX_CPU_NUMBER + 1];
  BLASLONG i, k0 = 0;
  for (i = 0; i < p; i++) {
    BLASLONG kw = (k - k0) / (p - i);
    jobs[i].args = *args;
    jobs[i].args.k = kw;
    jobs[i].args.nthreads = 1;
    jobs[i].args.a = (IFLOAT *)args->a + (transa ? k0 : k0 * args->lda);
    jobs[i].args.b = (IFLOAT *)args->b + (transb ? k0 * args->ldb : k0);
    if (i > 0) {
      jobs[i].args.c = buf + (i - 1) * m * n;
      jobs[i].args.ldc = m;
      jobs[i].args.beta = &zero;
    }
    jobs[i].fn = gemm[(transb << 2) | transa];
    queue[i].mode = mode;
    queue[i].routine = (void *)splitk_worker;
    queue[i].args = (blas_arg_t *)&jobs[i];
    queue[i].range_m = NULL;
    queue[i].range_n = NULL;
    queue[i].sa = NULL;
    queue[i].sb = NULL;
    queue[i].next = &queue[i + 1];
    k0 += kw;
  }
  queue[0].sa = sa;
  queue[0].sb = sb;
  queue[p - 1].next = NULL;
  exec_blas(p, queue);

  /* Reduce the p - 1 partial products into C, split over columns. */
  BLASLONG nr = MIN(p, n);
  jobs[0].args = *args;
  jobs[0].buf = buf;
  jobs[0].nbuf = p - 1;
  range[0] = 0;
  for (i = 0; i < nr; i++) {
    range[i + 1] = range[i] + (n - range[i]) / (nr - i);
    queue[i].mode = mode;
    queue[i].routine = (void *)splitk_reduce;
    queue[i].args = (blas_arg_t *)&jobs[0];
    queue[i].range_m = NULL;
    queue[i].range_n = &range[i];
    queue[i].sa = NULL;
    queue[i].sb = NULL;
    queue[i].next = &queue[i + 1];
  }
  queue[0].sa = sa;
  queue[0].sb = sb;
  queue[nr - 1].next = NULL;
  exec_blas(nr, queue);
  free(buf);
  return 1;
}
#endif

OPENBLAS_EXPORT
'''
anchor = "OPENBLAS_EXPORT\n"
idx = s.index(anchor)
s = s[:idx] + helper.lstrip("\n") + s[idx + len(anchor):]

call_old = "\n if (args.nthreads == 1) {\n#endif\n"
assert s.count(call_old) == 1, s.count(call_old)
call_new = """
#ifdef GEMM_SPLITK
 if (args.nthreads > 1 && gemm_splitk(&args, transa, transb, sa, sb)) {
 } else
#endif
 if (args.nthreads == 1) {
#endif
"""
s = s.replace(call_old, call_new)
path.write_text(s)
print("patched", path)
