#!/bin/bash
# In the pixi env, under benchlock. Positive control: probes/race.pyx under TSan, per Cython tree. Only `ctl` should report.
W=$(cd "$(dirname "$0")" && pwd); cd "$W"
E=$CONDA_PREFIX/lib
inc=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
suffix=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
mkdir -p tool
[ -f tool/libarcher.so ] || cp "$(ls "$E"/libarcher.so* | head -1)" tool/libarcher.so
rm -rf ../cy/ctl
cp -r ../cy/fixB2 ../cy/ctl
sed -i 's/nowait = self.acquire_gil or self.is_parallel or self.ends_parallel_block()/nowait = True/' ../cy/ctl/Cython/Compiler/Nodes.py
for tree in 331 fixA fixB2 ctl; do
  d=$W/probe-$tree; rm -rf "$d"; mkdir -p "$d"; cp race.pyx "$d/"
  cd "$d"
  PYTHONPATH=$W/../cy/$tree python "$W/../cy/$tree/cython.py" -3 race.pyx -o race.c > /dev/null 2>&1
  grep -h 'pragma omp for' race.c | sed 's/^ *//' | sort | uniq -c
  clang -O1 -g -w -fopenmp -fsanitize=thread -fPIC -shared -I"$inc" -I"$CONDA_PREFIX/include" -L"$E" -Wl,-rpath,"$E" race.c -o "race$suffix"
  env OMP_TOOL_LIBRARIES=$W/tool/libarcher.so TSAN_OPTIONS="ignore_noninstrumented_modules=1 exitcode=0 log_path=$d/tsan history_size=4" \
    LD_LIBRARY_PATH=$E LD_PRELOAD=$E/libclang_rt.tsan-x86_64.so:$E/libomp.so \
    timeout 120 python -I -c "import sys; sys.path.insert(0, '.'); import race; print('after_loop', [race.after_loop(2000, 4) for _ in range(5)]); print('between_loops', [race.between_loops(2000, 4) for _ in range(5)])" 2>&1 | tail -3
  echo "$tree exit=${PIPESTATUS[0]} reports=$(cat tsan.* 2>/dev/null | grep -c 'WARNING: ThreadSanitizer')"
  cat tsan.* 2>/dev/null | grep -E '^SUMMARY' | sed -E 's#^SUMMARY: ThreadSanitizer: ##; s# /[^ ]*/# #' | cut -c1-110 | sort | uniq -c | sort -rn | head -5
  cd "$W"
done
