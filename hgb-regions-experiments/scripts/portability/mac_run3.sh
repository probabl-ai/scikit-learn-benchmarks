#!/bin/bash
# Run under benchlock on mac-m2: conda clang and clang++ with libomp, Xcode's SDK.
export PATH=$HOME/.pixi/bin:$PATH
cd ~/work/hgb-port
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
export SDKROOT=$SDK CXX=clang++ CXXFLAGS="-fopenmp -isysroot $SDK"
pixi install 2>&1 | tail -1
rm -rf skl-conda-clang/build
TEST_TAIL=14 pixi run bash ./build_test.sh conda-clang clang "-fopenmp -isysroot $SDK"
S=venv-conda-clang/lib/python3.13/site-packages/sklearn/ensemble/_hist_gradient_boosting
nm -u $S/splitting.cpython-313-darwin.so | grep -E "kmpc" | head -3
otool -L $S/splitting.cpython-313-darwin.so | grep -i omp
echo "### mac3 done"
