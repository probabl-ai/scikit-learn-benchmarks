#!/bin/bash
# In the pixi env, under benchlock. The control tree is Fix B with nowait on every prange, so it must race.
W=$(cd "$(dirname "$0")" && pwd); cd "$W"
main=d5a61d1149cb; fused=9ade78bf1a25
rm -rf ../cy/ctl
cp -r ../cy/fixB2 ../cy/ctl
sed -i 's/nowait = self.acquire_gil or self.is_parallel or self.ends_parallel_block()/nowait = True/' ../cy/ctl/Cython/Compiler/Nodes.py
grep -n 'nowait = ' ../cy/ctl/Cython/Compiler/Nodes.py
python -c 'import sysconfig; print(sysconfig.get_paths()["include"])'
for spec in "m331 $main 331" "mA $main fixA" "mB $main fixB2" "fB $fused fixB2" "fctl $fused ctl"; do
  bash build.sh $spec 2>&1 | tail -8 | cut -c1-170
done
bash run.sh m331
bash run.sh mA
bash run.sh mB
bash run.sh fB SKLEARN_HGB_EXP=fuse3 SKLEARN_HGB_GUARD=pr
bash run.sh fctl SKLEARN_HGB_EXP=fuse3 SKLEARN_HGB_GUARD=pr
echo "# all done"
