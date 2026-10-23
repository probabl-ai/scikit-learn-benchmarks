import pytest

from sklbench.reporting.envs import _openmp_runtime_name

KMP_DUMP = {"_OPENMP": "201611", "[host] KMP_BLOCKTIME": "200"}
GOMP_DUMP = {"_OPENMP": "201511", "GOMP_SPINCOUNT": "300000"}


def _env(dump, *openmp_prefixes):
    pools = [{"user_api": "blas", "prefix": "libscipy_openblas"}]
    pools += [{"user_api": "openmp", "prefix": prefix} for prefix in openmp_prefixes]
    return {"openmp_runtime_info": dump, "threadpool_info": pools}


@pytest.mark.parametrize(
    "env, expected",
    [
        (_env(KMP_DUMP, "libomp", "libomp"), "LLVM OpenMP"),
        (_env(KMP_DUMP, "libomp", "vcomp"), "LLVM OpenMP"),
        (_env(KMP_DUMP, "libiomp"), "Intel OpenMP"),
        # Both loaded: the dump doesn't say which one printed it
        (_env(KMP_DUMP, "libomp", "libiomp"), "Intel/LLVM OpenMP"),
        (_env(KMP_DUMP), "Intel/LLVM OpenMP"),
        (_env(GOMP_DUMP, "libgomp"), "GNU libgomp"),
    ],
)
def test_openmp_runtime_name(env, expected):
    assert _openmp_runtime_name(env) == expected
