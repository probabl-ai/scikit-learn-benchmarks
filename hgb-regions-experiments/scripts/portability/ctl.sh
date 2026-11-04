#!/bin/bash
cd $WORK
for L in nbrace nbok; do bash tsan_build.sh $L hgb/fewer-regions-nb 2>&1 | tail -3 | cut -c1-150; bash tsan_run.sh $L SKLEARN_HGB_EXP=fuse3 SKLEARN_HGB_GUARD=pr; done
echo "### control done"
