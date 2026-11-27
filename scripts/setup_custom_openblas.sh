#!/usr/bin/env bash
# Build OpenBLAS at the commits pinned below and make a Pixi env use it.
# usage: scripts/setup_custom_openblas.sh sklearn-cf-develop-openblas|sklearn-cf-optimized-openblas
#
# Uses a few minutes of all the cores on the first run for a pin, so run it under
# benchlock. run.sh calls it for these envs.
set -euo pipefail

# Upstream develop.
BASE=c2fc6cf5f784aab8eea95de3876d12f546cf787f
# Heads of cakedev0/OpenBLAS PRs #3, #8, #10 and #11.
OPTIMIZED_MERGES="89e62160fdc721430bb369e221c7b49e81ee7a82 d851ca85f6fec94b3ec847e0e337e240de7d9109 2b66eeb857351aa5e1f4d5c5e44e74dcce99df55 21cbb8fc2e84e472c0f6f3c0adbff0c81a44785e"

env_name="${1:?usage: $0 PIXI_ENV}"
case "$env_name" in
    sklearn-cf-develop-openblas)
        merges=""
        label="develop ${BASE:0:8}"
        ;;
    sklearn-cf-optimized-openblas)
        merges="$OPTIMIZED_MERGES"
        label="develop ${BASE:0:8} + cakedev0/OpenBLAS #3, #8, #10, #11"
        ;;
    *)
        echo "error: no custom OpenBLAS for env '$env_name'" >&2
        exit 2
        ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="$root/.pixi/custom-openblas"
key="$(echo "$BASE $merges" | sha1sum | cut -c1-12)"
prefix="$cache/builds/$key"
src="$cache/src"

# The conda-forge feedstock flags, so that only the OpenBLAS commit differs from
# sklearn-cf-libgomp-openblas.
FLAGS="DYNAMIC_ARCH=1 BINARY=64 NO_LAPACK=0 TARGET=PRESCOTT BUILD_BFLOAT16=1 NO_AFFINITY=1 USE_THREAD=1 NUM_THREADS=128 USE_OPENMP=0 INTERFACE64=0"

if [ ! -e "$prefix/lib/libopenblas.so.0" ]; then
    mkdir -p "$cache"
    [ -d "$src" ] || git clone -q https://github.com/OpenMathLib/OpenBLAS.git "$src"
    git -C "$src" fetch -q origin "$BASE"
    git -C "$src" checkout -q -f --detach "$BASE"
    git -C "$src" clean -qfdx
    for commit in $merges; do
        git -C "$src" fetch -q https://github.com/cakedev0/OpenBLAS.git "$commit"
        git -C "$src" -c user.name=sklbench -c user.email=sklbench@example.invalid \
            merge -q --no-edit "$commit"
    done
    mkdir -p "$prefix"
    # Plain `make`: parallel `libs netlib shared` targets race on `ar` and
    # corrupt the archive.
    pixi exec -s gcc_linux-64=15 -s gfortran_linux-64=15 -s make -- bash -c "
        set -eu
        cd '$src'
        export FFLAGS=\"\${FFLAGS:-} -frecursive\"
        make -j$(nproc) $FLAGS CC=\$CC FC=\$FC HOSTCC=\$CC > '$prefix/build.log' 2>&1
        make $FLAGS CC=\$CC FC=\$FC HOSTCC=\$CC PREFIX='$prefix' install >> '$prefix/build.log' 2>&1
    "
fi

pixi install --frozen -e "$env_name"
lib="$root/.pixi/envs/$env_name/lib"
# A new file and new symlinks: the env's libopenblasp-r*.so is hard-linked into the
# pixi cache, so writing to it would change every env on the machine.
cp -f "$(readlink -f "$prefix/lib/libopenblas.so.0")" "$lib/libopenblas-custom.so"
for name in libblas.so libblas.so.3 libcblas.so libcblas.so.3 liblapack.so liblapack.so.3 \
    libopenblas.so libopenblas.so.0; do
    ln -sf libopenblas-custom.so "$lib/$name"
done

merged_json=""
for commit in $merges; do merged_json+="\"$commit\","; done
mkdir -p "$lib/../share"
cat > "$lib/../share/custom-openblas.json" <<EOF
{"label": "$label", "base": "$BASE", "merged": [${merged_json%,}]}
EOF

# Fails if numpy didn't load the build, or if a second OpenBLAS got loaded.
(cd "$root" && pixi run --frozen -e "$env_name" python -c '
import numpy, scipy.linalg, threadpoolctl
blas = [i for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"]
assert len(blas) == 1 and blas[0]["filepath"].endswith("libopenblas-custom.so"), blas
print("custom OpenBLAS:", blas[0]["version"], blas[0]["filepath"])
')
