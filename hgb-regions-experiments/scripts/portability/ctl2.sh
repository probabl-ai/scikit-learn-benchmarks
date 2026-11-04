#!/bin/bash
cd $WORK
bash tsan_build.sh nbrace hgb/fewer-regions-nb 2>&1 | tail -2 | cut -c1-150
timeout 900 bash tsan_run.sh nbrace SKLEARN_HGB_EXP=fuse3 SKLEARN_HGB_GUARD=pr
echo "### control2 done"
