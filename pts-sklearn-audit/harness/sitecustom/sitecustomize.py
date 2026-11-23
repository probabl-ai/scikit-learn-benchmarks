import os

# In a benchlock socket slot the process is bound to half the CPUs, but
# os.cpu_count() still reports all of them (bench_hist_gradient_boosting_threading
# uses it as its thread count). Make it agree with the affinity mask, as on a
# machine of that size.
os.cpu_count = lambda: len(os.sched_getaffinity(0))

# shim/python3 attaches py-spy to this process, which is not py-spy's child:
# with kernel.yama.ptrace_scope=1 the process has to allow it.
_ready = os.environ.pop("PTS_AUDIT_READY_FILE", None)
if _ready:
    import ctypes

    PR_SET_PTRACER, PR_SET_PTRACER_ANY = 0x59616D61, ctypes.c_ulong(-1)
    ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    open(_ready, "w").close()
