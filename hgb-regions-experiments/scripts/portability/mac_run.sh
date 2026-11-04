#!/bin/bash
# Run under benchlock on mac-m2. The Command Line Tools' macOS 27 SDK doesn't work with this
# Xcode 16 linker, so use Xcode's SDK.
export PATH=$HOME/.pixi/bin:$PATH
cd ~/work/hgb-port
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
export SDKROOT=$SDK
rm -rf skl-conda-clang/build skl-apple-noomp/build
TEST_TAIL=14 pixi run bash ./build_test.sh conda-clang clang "-fopenmp -isysroot $SDK"
TEST_TAIL=14 pixi run bash ./build_test.sh apple-noomp /usr/bin/clang "-isysroot $SDK"
echo "### mac done"
