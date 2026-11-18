"""Build the OpenMP kernel of `faiss_emeans_cpu.FastNumpyEmeans` in place:
``pixi run -e faiss-cpu build-emeans-kernel``."""

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    extension = Extension(
        "sklbench.runners.estimator.wrappers._emeans_kernel",
        [str(HERE / "_emeans_kernel.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
    setup(
        name="sklbench-emeans-kernel",
        ext_modules=cythonize([extension], quiet=True),
        script_args=[
            "build_ext",
            "--inplace",
            "--build-temp",
            str(HERE / "build"),
            "--build-lib",
            str(HERE / "build"),
        ],
    )
