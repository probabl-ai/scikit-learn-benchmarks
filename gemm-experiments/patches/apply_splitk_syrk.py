"""Add a split-K path to OpenBLAS interface/syrk.c (prototype).

usage: python apply_splitk_syrk.py OPENBLAS_SRC

Same idea as apply_splitk_gemm.py for C = alpha op(A) op(A)^T + beta C,
i.e. X.T @ X: when n is small for the team (n * n < nthreads * 128^2) and k
is long, split k into one chunk per thread, run the single-threaded driver
on each chunk (chunk 0 into C with the user's beta, the others into private
n x n buffers with beta = 0) and add the uplo triangle of the buffers into
C, split over columns by equal triangle area. OPENBLAS_SYRK_SPLITK=0
disables it, =N forces N chunks (read once).
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "interface" / "syrk.c"
s = path.read_text()
assert "SYRK_SPLITK" not in s, "already patched"

helper = r'''
#if defined(SMP) && !defined(COMPLEX) && !defined(XDOUBLE) && !defined(USE_SIMPLE_THREADED_LEVEL3)
#define SYRK_SPLITK 1
#include <math.h>

struct syrk_splitk_job {
  blas_arg_t args;  /* first member: exec_blas passes &job as the args */
  int (*fn)(blas_arg_t *, BLASLONG *, BLASLONG *, FLOAT *, FLOAT *, BLASLONG);
  FLOAT *buf;
  BLASLONG nbuf;
  int uplo;
};

static int syrk_splitk_env = -2;

static BLASLONG syrk_splitk_nchunks(BLASLONG n, BLASLONG k, int nthreads) {
  if (syrk_splitk_env == -2) {
    char *e = getenv("OPENBLAS_SYRK_SPLITK");
    syrk_splitk_env = e ? atoi(e) : -1;
  }
  if (syrk_splitk_env == 0 || nthreads < 2) return 1;
  if (syrk_splitk_env > 0) return MIN(syrk_splitk_env, nthreads);
  BLASLONG p = nthreads;
  if (n * n >= (BLASLONG)nthreads * 128 * 128) return 1;
  if (k / p < 512) p = k / 512;
  BLASLONG cap = (64L << 20) / (n * n * (BLASLONG)sizeof(FLOAT));
  if (p - 1 > cap) p = cap + 1;
  if (2 * p < nthreads) return 1;
  return p < 2 ? 1 : p;
}

static int syrk_splitk_worker(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                              FLOAT *sa, FLOAT *sb, BLASLONG mypos) {
  struct syrk_splitk_job *job = (struct syrk_splitk_job *)args;
  return job->fn(&job->args, NULL, NULL, sa, sb, 0);
}

static int syrk_splitk_reduce(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                              FLOAT *sa, FLOAT *sb, BLASLONG mypos) {
  struct syrk_splitk_job *job = (struct syrk_splitk_job *)args;
  BLASLONG n = job->args.n, ldc = job->args.ldc;
  FLOAT *c = (FLOAT *)job->args.c;
  for (BLASLONG j = range_n[0]; j < range_n[1]; j++) {
    BLASLONG i0 = job->uplo ? j : 0, i1 = job->uplo ? n : j + 1;
    FLOAT *cj = c + j * ldc;
    for (BLASLONG p = 0; p < job->nbuf; p++) {
      FLOAT *bj = job->buf + p * n * n + j * n;
      for (BLASLONG i = i0; i < i1; i++) cj[i] += bj[i];
    }
  }
  return 0;
}

static int syrk_splitk(blas_arg_t *args, int uplo, int trans, FLOAT *sa, FLOAT *sb) {
  BLASLONG n = args->n, k = args->k;
  if (((FLOAT *)args->alpha)[0] == ZERO) return 0;
  BLASLONG p = syrk_splitk_nchunks(n, k, args->nthreads);
  if (p < 2) return 0;
  FLOAT *buf = (FLOAT *)malloc((size_t)(p - 1) * n * n * sizeof(FLOAT));
  if (buf == NULL) return 0;
  static FLOAT zero = ZERO;
#ifdef DOUBLE
  int mode = BLAS_DOUBLE | BLAS_REAL;
#else
  int mode = BLAS_SINGLE | BLAS_REAL;
#endif
  struct syrk_splitk_job jobs[MAX_CPU_NUMBER];
  blas_queue_t queue[MAX_CPU_NUMBER];
  BLASLONG range[MAX_CPU_NUMBER + 1];
  BLASLONG i, k0 = 0;
  for (i = 0; i < p; i++) {
    BLASLONG kw = (k - k0) / (p - i);
    jobs[i].args = *args;
    jobs[i].args.k = kw;
    jobs[i].args.nthreads = 1;
    /* N: A is n x k, chunk = columns k0.. ; T: A is k x n, chunk = rows k0.. */
    jobs[i].args.a = (FLOAT *)args->a + (trans ? k0 : k0 * args->lda);
    if (i > 0) {
      jobs[i].args.c = buf + (i - 1) * n * n;
      jobs[i].args.ldc = n;
      jobs[i].args.beta = &zero;
    }
    jobs[i].fn = syrk[(uplo << 1) | trans];
    queue[i].mode = mode;
    queue[i].routine = (void *)syrk_splitk_worker;
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

  /* Column ranges with equal triangle area: column j of the upper triangle
     holds j + 1 entries, of the lower n - j. */
  BLASLONG nr = MIN(p, n);
  jobs[0].args = *args;
  jobs[0].buf = buf;
  jobs[0].nbuf = p - 1;
  jobs[0].uplo = uplo;
  range[0] = 0;
  for (i = 1; i < nr; i++) {
    double f = uplo ? 1.0 - sqrt(1.0 - (double)i / nr) : sqrt((double)i / nr);
    range[i] = MAX(range[i - 1], MIN(n, (BLASLONG)(f * n)));
  }
  range[nr] = n;
  for (i = 0; i < nr; i++) {
    queue[i].mode = mode;
    queue[i].routine = (void *)syrk_splitk_reduce;
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

call_old = "\n  if (args.nthreads == 1) {\n#endif\n"
assert s.count(call_old) == 1, s.count(call_old)
s = s.replace(call_old, """
#ifdef SYRK_SPLITK
  if (args.nthreads > 1 && syrk_splitk(&args, uplo, trans, sa, sb)) {
  } else
#endif
  if (args.nthreads == 1) {
#endif
""")
path.write_text(s)
print("patched", path)
