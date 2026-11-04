#!/bin/bash
# From this laptop: copy the Cython trees and the scripts to a host's private work dir. usage: send_sources.sh HOST
set -e
h=$1
ssh $h 'mkdir -p ~/work/cython-prange/cy'
for pair in 324=3.2.4 331=3.3.1 fixA=prange-fix-end-barrier fixB2=prange-no-start-barrier; do
  n=${pair%%=*}; rev=${pair#*=}
  git -C ~/cython-prange archive $rev | ssh $h "rm -rf ~/work/cython-prange/cy/$n && mkdir -p ~/work/cython-prange/cy/$n && tar -x -C ~/work/cython-prange/cy/$n"
done
tar -C "$(dirname "$0")" -c build.sh other_case.py run_ab.sh other_ab.py sitecustomize.py bench_hgb.py host_all.sh sklearn_tests.sh | ssh $h "tar -x -C ~/work/cython-prange"
