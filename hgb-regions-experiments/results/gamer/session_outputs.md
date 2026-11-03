# Raw outputs copied from the session transcript

Outputs that were printed to the terminal and not saved on gamer, in run order (UTC).

## 2026-11-03T16:16:37.710Z

```
== libomp_main
env=sklearn-dev-libomp sklearn-src=b8de23fabf FIX Fix HGB binning on features with only missing values (#35216)
POOL openmp openmp /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1.0.0 24
POOL blas openblas /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libopenblasp-r0.3.34.so 24
16
== gomp_main
env=sklearn-dev sklearn-src=1420ffd397 MNT Remove deprecated parameters in Displays for 1.10 (#35276)
POOL openmp openmp /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1.0.0 24
POOL blas openblas /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libopenblasp-r0.3.34.so 24
16
== gomp_pr
env=sklearn-dev sklearn-src=313be4c8c8 FIX Pass the detected wait policy to subprocesses via os.environ
POOL openmp openmp /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1.0.0 24
POOL blas openblas /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libopenblasp-r0.3.34.so 24
9
==> build_main.log <==
Run benchmarks with:

  pixi run -e 'sklearn-dev' python -m sklbench --config configs/smoke_check_test.py

==> build_pr.log <==
Run benchmarks with:

  pixi run -e 'sklearn-dev' python -m sklbench --config configs/smoke_check_test.py
{"label": "omp24_n24", "omp": 24, "n": "", "rc": 0, "tree_n_threads": 23, "fit": [19.85, 20.93, 21.38], "hist_time": 17.42, "find_split_time": 1.91, "apply_split_time": 1.1, "grower_init_time": 0.3}
{"label": "omp24_n23", "omp": 24, "n": "23", "rc": 0, "tree_n_threads": 23, "fit": [19.72, 20.91, 20.84], "hist_time": 17.35, "find_split_time": 1.87, "apply_split_time": 1.07, "grower_init_time": 0.3}
{"label": "omp23_nA", "omp": 23, "n": "", "rc": 0, "tree_n_threads": 23, "fit": [20.84, 20.21, 20.96], "hist_time": 17.26, "find_split_time": 1.93, "apply_split_time": 1.1, "grower_init_time": 0.3}
{"label": "omp24_n22", "omp": 24, "n": "22", "rc": 0, "tree_n_threads": 22, "fit": [19.3, 17.34, 18.66], "hist_time": 15.23, "find_split_time": 1.89, "apply_split_time": 1.08, "grower_init_time": 0.29}
{"label": "omp24_n20", "omp": 24, "n": "20", "rc": 0, "tree_n_threads": 20, "fit": [15.69, 13.25, 14.04], "hist_time": 10.72, "find_split_time": 1.8, "apply_split_time": 1.0, "grower_init_time": 0.25}
{"label": "omp24_n16", "omp": 24, "n": "16", "rc": 0, "tree_n_threads": 16, "fit": [4.26, 5.0, 4.87], "hist_time": 2.72, "find_split_time": 1.11, "apply_split_time": 0.64, "grower_init_time": 0.24}
{"label": "omp16_nA", "omp": 16, "n": "", "rc": 0, "tree_n_threads": 15, "fit": [5.06, 5.33, 4.86], "hist_time": 2.77, "find_split_time": 1.14, "apply_split_time": 0.68, "grower_init_time": 0.24}
{"label": "omp24_n23_spin", "omp": 24, "n": "23", "rc": 0, "tree_n_threads": 23, "fit": [1.87, 2.82, 1.72], "hist_time": 1.03, "find_split_time": 0.21, "apply_split_time": 0.21, "grower_init_time": 0.15}
{"label": "omp24_n24", "omp": 24, "n": "", "rc": 0, "tree_n_threads": 23, "fit": [21.45, 20.02, 20.71], "hist_time": 17.29, "find_split_time": 1.82, "apply_split_time": 1.1, "grower_init_time": 0.3}
```

## 2026-11-03T16:25:36.238Z

```
benchlock: waiting for gkdal11989 all, held by: alacote@gkdal11989 pid=827908 since=2026-11-03T14:21:32+0000 slot=all tag=gemm: syrk PR v4 vs develop cmd=env BUILD=sr-syrk4 OUT=syrk_v4 SMALL_MACHINE=1 PIN=compact ROUNDS=3 scripts/run_syrk_v3.sh gamer 2,4,8,12,16,24
== queue done 2026-11-03T16:21:15+00:00
{"label": "omp24_n24", "omp": 24, "n": "", "rc": 0, "tree_n_threads": 23, "fit": [21.45, 20.02, 20.71], "hist_time": 17.29, "find_split_time": 1.82, "apply_split_time": 1.1, "grower_init_time": 0.3}
{"label": "omp24_n23", "omp": 24, "n": "23", "rc": 0, "tree_n_threads": 23, "fit": [21.37, 20.85, 20.81], "hist_time": 17.43, "find_split_time": 1.92, "apply_split_time": 1.09, "grower_init_time": 0.3}
{"label": "omp23_nA", "omp": 23, "n": "", "rc": 0, "tree_n_threads": 23, "fit": [20.93, 20.85, 20.82], "hist_time": 17.35, "find_split_time": 1.93, "apply_split_time": 1.09, "grower_init_time": 0.29}
{"label": "omp24_n22", "omp": 24, "n": "22", "rc": 0, "tree_n_threads": 22, "fit": [20.13, 18.48, 19.05], "hist_time": 15.62, "find_split_time": 1.86, "apply_split_time": 1.08, "grower_init_time": 0.3}
{"label": "omp24_n20", "omp": 24, "n": "20", "rc": 0, "tree_n_threads": 20, "fit": [14.64, 17.16, 14.4], "hist_time": 11.49, "find_split_time": 1.68, "apply_split_time": 0.99, "grower_init_time": 0.26}
{"label": "omp24_n16", "omp": 24, "n": "16", "rc": 0, "tree_n_threads": 16, "fit": [4.42, 4.22, 4.6], "hist_time": 2.55, "find_split_time": 0.87, "apply_split_time": 0.6, "grower_init_time": 0.23}
{"label": "omp16_nA", "omp": 16, "n": "", "rc": 0, "tree_n_threads": 15, "fit": [4.37, 4.81, 4.49], "hist_time": 2.6, "find_split_time": 0.96, "apply_split_time": 0.58, "grower_init_time": 0.23}
{"label": "omp24_n23_spin", "omp": 24, "n": "23", "rc": 0, "tree_n_threads": 23, "fit": [1.85, 1.97, 2.52], "hist_time": 1.07, "find_split_time": 0.21, "apply_split_time": 0.19, "grower_init_time": 0.14}
gkdal11989:
  all: held by alacote@gkdal11989 pid=1309659 since=2026-11-03T16:21:33+0000 slot=all tag=HGB PR #34935 year_prediction_msd wait-policy check cmd=/home/alacote/work/gamer-run/msd/run_msd2.sh
sklearn-src=313be4c8c8 FIX Pass the detected wait policy to subprocesses via os.environ
POOL openmp openmp /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1.0.0 24
POOL blas openblas /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libopenblasp-r0.3.34.so 24
POOL blas openblas /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libopenblasp-r0.3.34.so 24
POOL openmp openmp /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so 24
{"env": "sklearn-dev", "label": "pr_default", "omp": 24, "n": "", "rc": 0, "tree_n_threads": 23, "fit": [17.15, 16.43, 17.49], "hist_time": 13.62, "find_split_time": 1.98, "apply_split_time": 1.04, "grower_init_time": 0.26}
{"env": "sklearn-dev", "label": "pr_policy_active", "omp": 24, "n": "", "rc": 0, "tree_n_threads": 23, "fit": [4.63, 4.26, 5.08], "hist_time": 2.46, "find_split_time": 1.05, "apply_split_time": 0.73, "grower_init_time": 0.21}
{"env": "sklearn-dev", "label": "pr_policy_passive", "omp": 24, "n": "", "rc": 0, "tree_n_threads": 23, "fit": [16.27, 15.06, 16.94], "hist_time": 12.97, "find_split_time": 1.96, "apply_split_time": 1.0, "grower_init_time": 0.25}
{"env": "sklearn-dev", "label": "pr_n18", "omp": 24, "n": "18", "rc": 0, "tree_n_threads": 18, "fit": [4.96, 5.35, 5.38], "hist_time": 2.99, "find_split_time": 1.41, "apply_split_time": 0.57, "grower_init_time": 0.21}
```

## 2026-11-03T16:38:54.561Z

```
sklearn-dev          pr_default         threads=23 fit_med= 17.15 hist= 13.62 find_split= 2.00 apply_split= 1.06 n=3
sklearn-dev          pr_n18             threads=18 fit_med=  5.35 hist=  2.99 find_split= 1.41 apply_split= 0.57 n=3
sklearn-dev          pr_policy_active   threads=23 fit_med=  5.08 hist=  2.64 find_split= 1.22 apply_split= 0.83 n=3
sklearn-dev          pr_policy_passive  threads=23 fit_med= 16.70 hist= 12.98 find_split= 1.98 apply_split= 1.04 n=3
sklearn-dev-libomp   pr_default         threads=23 fit_med= 16.48 hist= 12.89 find_split= 1.99 apply_split= 1.02 n=3
sklearn-dev-libomp   pr_n18             threads=18 fit_med=  5.67 hist=  3.10 find_split= 1.50 apply_split= 0.68 n=3
sklearn-dev-libomp   pr_policy_active   threads=23 fit_med=  4.85 hist=  2.52 find_split= 1.13 apply_split= 0.78 n=3
sklearn-dev-libomp   pr_policy_passive  threads=23 fit_med= 16.68 hist= 13.18 find_split= 1.99 apply_split= 1.03 n=3
== msd2 done 2026-11-03T16:37:05+00:00
gkdal11989:
  all: free
```

## 2026-11-03T17:24:01.084Z

```
38905644
== libgomp passive (default)
T=23 item_us=50 ideal_region_us=200
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      575.4      639.8        335.9        401.7          232.0          7         20.0
      20      562.3      691.3        217.9        372.9          231.2          4         25.1
      50      528.6      660.2        203.1        345.7          230.9          7         27.3
     100      502.6      609.9        179.5        275.2          230.2          4         36.7
     160      506.2      585.4        162.1        223.6          231.0          0         44.5
     300      401.7      535.2        152.7        179.6          231.7          0         44.5
     500      430.1      597.0        154.2        190.4          235.0          0         52.8
    1000      607.7      685.9        201.4        240.0          312.2          0         57.0
    2000      636.9      714.2        217.3        254.3          329.7          0         52.3
== libgomp passive (default)
T=16 item_us=50 ideal_region_us=300
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      433.1      532.8        128.8        224.1          297.2         69          5.6
      20      622.1      749.8        199.5        333.9          333.5         16         22.6
      50      622.0      726.5        195.6        305.9          334.8          3         27.5
     100      612.9      687.5        165.5        247.3          335.0          1         27.4
     160      600.2      671.7        161.9        203.9          335.6          0         29.3
     300      510.9      643.2        148.2        188.7          336.3          0         33.2
     500      512.3      648.6        152.9        192.4          337.2          0         34.9
    1000      608.1      705.1        165.4        200.8          391.4          0         39.2
    2000      587.6      693.5        159.4        195.8          388.5          0         39.8
== libgomp passive (default)
T=8 item_us=50 ideal_region_us=600
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      845.3      928.1        122.8        246.8          625.5         42          8.4
      20      836.5      916.2        138.4        221.7          630.3         29          8.1
      50      829.0      903.5        121.8        189.7          632.8         22          9.0
     100      811.6      894.6        113.2        155.9          634.2         12         11.2
     160      781.2      890.6        111.8        148.7          634.1          4         13.7
     300      761.7      880.9        106.4        139.8          632.2          4         10.7
     500      761.7      882.0        109.9        145.8          634.3          0         15.3
    1000      833.5      935.0        128.4        159.6          670.2          0         18.7
    2000      815.1      917.8        119.9        147.3          667.8          0         19.0
== libgomp spinning (GOMP_SPINCOUNT=300000 KMP_BLOCKTIME=200ms)
T=23 item_us=50 ideal_region_us=200
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      209.5      210.4          1.0          1.2          207.8         24          0.0
      20      209.3      210.3          0.9          1.1          207.8         25          0.0
      50      209.4      210.3          0.9          1.1          207.8         25          0.0
     100      209.4      210.2          0.9          1.1          207.8         22          0.0
     160      209.4      210.2          0.9          1.1          207.8         21          0.1
     300      209.4      210.3          1.0          1.1          207.8         21          0.1
     500      209.5      210.3          1.0          1.2          207.8         22          0.1
    1000      209.5      210.4          1.1          1.2          207.8         20          0.2
    2000      209.5      210.4          1.1          1.2          207.8         21          0.4
== libomp passive (default)
T=23 item_us=50 ideal_region_us=200
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      460.4      586.5        178.2        274.0          228.6          5         17.3
      20      532.4      675.4        235.8        349.6          230.0          6         19.0
      50      497.5      646.7        201.4        322.8          229.6          5         17.8
     100      400.9      614.1        157.4        292.3          229.4          2         20.2
     160      374.1      406.2        140.5        172.2          227.6          1         18.4
     300      383.7      412.6        148.1        177.0          229.0          0         24.6
     500      457.1      540.5        150.7        188.9          231.3          0         29.5
    1000      511.0      644.7        185.4        215.3          301.4          0         31.1
    2000      528.1      629.7        187.6        221.8          310.7         85         31.4
== libomp passive (default)
T=16 item_us=50 ideal_region_us=300
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      470.0      566.3        133.1        194.4          315.6          9          7.7
      20      482.4      581.1        139.4        183.7          320.3          2          8.6
      50      317.3      462.0          8.4        126.4          303.4          5          5.8
     100      454.3      574.3        120.2        155.7          318.1         12         12.4
     160      467.3      574.6        127.0        155.5          322.7          0         14.1
     300      506.8      590.1        131.7        155.7          324.3          0         18.2
     500      494.5      568.5        128.1        153.8          325.2          0         21.0
    1000      592.2      696.4        186.6        236.8          382.0          0         20.6
    2000      621.2      737.6        194.1        231.2          386.3         22         21.9
== libomp passive (default)
T=8 item_us=50 ideal_region_us=600
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      823.3      937.0        132.8        224.8          617.0         14          6.1
      20      870.7     1017.8        172.7        260.8          626.1          3          6.7
      50      843.5      987.5        148.0        245.3          624.9          4          6.7
     100      838.1      983.9        133.9        235.8          625.4          4          6.1
     160      836.7      968.7        132.3        235.7          624.7          3          4.4
     300      873.3     1014.3        196.3        249.6          626.4          3          9.4
     500      953.3     1117.2        236.6        317.5          641.1          0         10.1
    1000      976.7     1109.1        251.0        332.7          666.9         55          9.2
    2000     1038.4     1117.6        295.9        327.1          656.7         59          8.7
== libomp spinning (GOMP_SPINCOUNT=300000 KMP_BLOCKTIME=200ms)
T=23 item_us=50 ideal_region_us=200
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      208.5      209.9          0.7          0.8          207.4         34          0.0
      20      208.5      209.6          0.7          0.8          207.4         34          0.0
      50      208.5      209.6          0.7          0.8          207.4         37          0.0
     100      208.5      209.6          0.7          0.8          207.4         38          0.0
     160      208.5      209.6          0.7          0.8          207.4         39          0.1
     300      211.5      212.8          0.7          0.8          210.5         42          0.1
     500      212.5      213.6          0.7          0.8          211.4         39          0.1
    1000      212.6      213.7          0.8          0.9          211.4         31          0.2
    2000      212.6      213.8          0.8          0.9          211.4         29          0.3
== check runtime under LD_PRELOAD

OPENMP DISPLAY ENVIRONMENT BEGIN
  _OPENMP = '202111'
== wake done 2026-11-03T17:24:01+00:00
```

## 2026-11-03T17:27:02.549Z

```
/usr/bin/perf
4
policy=default C3_entries=3581028 wall=53.38 user=117.20 sys=15.32 vol_cs=7346946 invol_cs=427 fits=[18.0, 15.7, 16.5]
policy=active C3_entries=550850 wall=19.10 user=100.50 sys=17.61 vol_cs=12941550 invol_cs=406 fits=[5.1, 5.7, 5.3]
policy=default C3_entries=3701696 wall=54.14 user=119.34 sys=15.26 vol_cs=7398074 invol_cs=449 fits=[17.2, 15.9, 18.0]
policy=active C3_entries=632984 wall=20.53 user=99.19 sys=18.41 vol_cs=12955349 invol_cs=494 fits=[5.1, 4.9, 7.3]
```

## 2026-11-03T17:29:05.018Z

```
== libgomp passive
T=23 item_kb=32 passes=4 per_thread_kb=128
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       24.4       27.7         11.6            3.7            4.3        0.0         3.0
    160      233.0      297.4        214.1            5.5           11.5        0.0        35.1
    500      181.1      230.7        155.8            9.1           12.6        0.0        40.5
   2000      353.2      455.0        219.0           13.2           33.0        0.1        45.1
== libgomp spin
T=23 item_kb=32 passes=4 per_thread_kb=128
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0        6.2        6.6          1.2            4.3            4.4        0.0         0.0
    160        6.2        7.6          1.3            4.2            4.4        0.0         0.0
    500        7.3        7.7          1.3            5.3            5.6        0.0         0.1
   2000        7.4        7.8          1.3            5.5            5.7        0.0         0.3
== libgomp passive
T=23 item_kb=256 passes=1 per_thread_kb=1024
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       46.5       49.0         10.9           20.9           29.6        0.0         0.0
    160      236.5      307.8        201.9           23.1           39.9        0.1        39.4
    500      195.4      245.5        154.8           28.8           40.3        0.0        43.7
   2000      371.1      484.1        193.6           54.0           88.6        0.1        44.5
== libgomp spin
T=23 item_kb=256 passes=1 per_thread_kb=1024
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       29.2       30.0          1.1           16.7           28.1        0.0         0.0
    160       31.8       33.0          1.1           19.5           30.6        0.0         0.0
    500       32.6       33.4          1.1           19.9           31.3        0.0         0.1
   2000       33.3       33.8          1.1           20.6           32.2        0.0         0.3
== passive, 16 threads
T=16 item_kb=32 passes=4 per_thread_kb=192
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       19.6       20.7          8.9            5.7            6.0        0.0         0.2
    160      158.1      283.1        118.3           14.3           17.5        0.0        22.4
    500      204.0      311.1        128.3           20.3           23.9        0.0        27.9
   2000      332.2      416.6        195.1           17.1           39.1        0.0        33.0
== wake2 done 2026-11-03T17:29:04+00:00
```

## 2026-11-03T17:38:19.487Z

```
5:c3() { cat /sys/devices/system/cpu/cpu*/cpuidle/state3/usage | paste -sd+ | bc; }; c2() { cat /sys/devices/system/cpu/cpu*/cpuidle/state2/usage | paste -sd+ | bc; }
9:  b=$(c3); b2=$(c2)
12:  a=$(c3); a2=$(c2)
13:  echo "policy=$pol C3_entries=$((a-b)) C2_entries=$((a2-b2)) $(tail -n 1 $D/c.err) fits=$(python3 -c "import json;print([round(json.loads(l)['time_ms']['fit']/1000,1) for l in open('$D/c.jsonl')])")"
C3 disable flags:  24 1
policy=default C3_entries=0 C2_entries=6044460 wall=61.21 user=129.28 sys=17.29 vol_cs=7573478 invol_cs=534 fits=[19.8, 19.7, 18.8]
policy=active C3_entries=0 C2_entries=7723192 wall=81.57 user=140.67 sys=31.28 vol_cs=13667925 invol_cs=393 fits=[26.5, 26.2, 25.9]
policy=default C3_entries=0 C2_entries=6135648 wall=63.57 user=127.76 sys=17.32 vol_cs=7570951 invol_cs=596 fits=[20.1, 20.8, 19.7]
policy=active C3_entries=0 C2_entries=8350140 wall=88.11 user=144.46 sys=32.43 vol_cs=13648554 invol_cs=360 fits=[28.6, 24.7, 31.7]
== wake2 libgomp passive, C3 off
T=23 item_kb=32 passes=4 per_thread_kb=128
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0      380.0      520.4        328.1           11.9           15.9        0.1         0.0
    160      300.3      384.6        171.0           11.4           15.7        0.0         0.0
    500      285.4      379.6        162.1           12.3           16.4        0.0         0.0
   2000      344.6      442.5        208.5           11.7           31.5        0.3         0.0
== c3off done 2026-11-03T17:38:19+00:00
```

## 2026-11-03T17:38:59.187Z

```
== spin
T=23 item_kb=32 passes=4 per_thread_kb=128
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0        5.9       10.5          0.9            4.1            4.2        0.0         0.0
    160        6.8        7.4          0.9            5.0            5.2        0.1         0.0
   2000        7.4        7.8          0.9            5.6            5.8        0.0         0.0
== passive
T=23 item_kb=32 passes=4 per_thread_kb=128
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0      393.5      523.0        341.5           11.3           13.3        0.3         0.0
    160      256.2      317.8        236.7            6.2           11.6        0.0         0.0
   2000      341.8      438.4        218.2           12.8           32.8        0.1         0.0
== passive 1KB items, compute-light
T=23 item_us=5 ideal_region_us=20
  gap_us region_med region_p90 wake_max_med wake_max_p90 slowest_busy_med slow_wake_E%  C3_per_iter
       0      402.7      593.5        282.3        417.0           51.7          0          0.0
     160      335.4      449.8        179.1        258.1           51.7          0          0.0
    2000      369.4      485.7        195.4        243.2          102.6          0          0.0
```

## 2026-11-03T20:06:50.580Z

```
#!/bin/bash
# One PR fit on year_prediction_msd (23 threads), serial vs parallel find_split, with OS counters.
cd ~/scikit-learn-benchmarks
D=~/work/gamer-run/msd
c3() { cat /sys/devices/system/cpu/cpu*/cpuidle/state3/usage | paste -sd+ | bc; }; c2() { cat /sys/devices/system/cpu/cpu*/cpuidle/state2/usage | paste -sd+ | bc; }
for rep in 1 2; do
for pol in default active; do
  extra=""; [ $pol = active ] && extra="SKLEARN_OMP_WAIT_POLICY=active"
  b=$(c3); b2=$(c2)
  /usr/bin/time -f "wall=%e user=%U sys=%S vol_cs=%w invol_cs=%c" env OMP_NUM_THREADS=24 $extra \
    pixi run --frozen -e sklearn-dev python $D/patched_runner.py --case-file $D/case.json --n-runs 3 --output-jsonl $D/c.jsonl 2> $D/c.err
  a=$(c3); a2=$(c2)
  echo "policy=$pol C3_entries=$((a-b)) C2_entries=$((a2-b2)) $(tail -n 1 $D/c.err) fits=$(python3 -c "import json;print([round(json.loads(l)['time_ms']['fit']/1000,1) for l in open('$D/c.jsonl')])")"
  rm -f $D/c.jsonl
done
done
----
#!/bin/bash
cd ~/work/gamer-run/wake
E=~/scikit-learn-benchmarks/.pixi/envs
$E/sklearn-dev/bin/gcc -O2 -fopenmp wake2.c -o wake2 || exit 1
GAPS="0 160 500 2000"
# ~per-thread working sets: 32 KB items -> 128 KB/thread (fits L2), 256 KB items -> 1 MB/thread
for cfg in "32 4" "256 1"; do
  set -- $cfg
  for mode in passive spin; do
    if [ $mode = spin ]; then X="GOMP_SPINCOUNT=300000"; else X=""; fi
    echo "== libgomp $mode"
    env $X OMP_NUM_THREADS=23 ./wake2 1000 $1 $2 $GAPS
  done
done
echo "== passive, 16 threads"
OMP_NUM_THREADS=16 ./wake2 1000 32 4 $GAPS
echo "== wake2 done $(date -Is)"
----
cat: wake/c3off.sh: No such file or directory
msd:
__pycache__
build_main.log
build_pr.log
c.err
c3off.sh
case.json
counters.sh
counters_c3off.sh
gomp_main.jsonl
gomp_main.log
gomp_pr.jsonl
gomp_pr.log
libomp_main.jsonl
libomp_main.log
msd2.jsonl
msd2.log
patched_runner.py
run_msd.sh
run_msd2.sh
tmp.err
tmp.jsonl

wake:
run_wake.sh
run_wake2.sh
wake
wake.c
wake2
wake2.c
```

## 2026-11-03T20:07:32.992Z

```
### rep=1 states=c1only disable=0 0 1 1
== wake2 passive 23T
T=23 item_kb=32 passes=4 per_thread_kb=128
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       22.8       61.3         12.2            3.7            3.8        0.2         0.0
    160       26.4       34.3         11.7            3.8            8.2        0.1         0.0
    500       46.9       53.9         20.8           11.6           17.1        0.1         0.0
   2000       50.8       53.7         21.2           11.9           20.8        0.0         0.0
policy=default C1=8184575 C2=0 C3=0 wall=15.36 user=88.45 sys=9.91 vol_cs=7015154 invol_cs=344 fits=[4.2, 4.1, 4.2]
```

## 2026-11-03T20:10:50.304Z

```
### rep=1 states=c1only disable=0 0 1 1
== wake2 passive 23T
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       22.8       61.3         12.2            3.7            3.8        0.2         0.0
    160       26.4       34.3         11.7            3.8            8.2        0.1         0.0
    500       46.9       53.9         20.8           11.6           17.1        0.1         0.0
--
### rep=1 states=all disable=0 0 0 0
== wake2 passive 23T
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       54.5      407.0         29.3            9.1           11.7        0.0        11.7
    160      221.5      283.2        203.7            5.7           11.6        0.0        36.5
    500      183.6      229.1        157.9            6.7           13.0        0.0        39.5
--
### rep=2 states=c1only disable=0 0 1 1
== wake2 passive 23T
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       22.9       59.5         12.2            3.7            3.8        0.0         0.0
    160       23.6       34.8         11.8            3.7            4.6        0.1         0.0
    500       46.2       51.6         20.3           11.7           17.2        0.0         0.0
--
### rep=2 states=all disable=0 0 0 0
== wake2 passive 23T
 gap_us region_med region_p90 wake_max_med   busy_med_thr   busy_max_thr  migrated% C3_per_iter
      0       38.3      373.2         15.6            9.4            9.6        0.0         9.4
    160      207.5      269.5        187.3            6.4           11.6        0.1        34.3
    500      183.1      234.2        157.9            8.8           12.6        0.1        41.3
  all: free
```

## 2026-11-03T20:25:05.086Z

```
== gomp passive (libgomp.so.1 => /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1)
const big=23 small=4  region_us(big) med=354.4 p90=483.1  distinct_tids=23
alt   big=23 small=4  region_us(big) med=288.5 p90=324.1  region_us(small) med=18.4  distinct_tids=19004
== gomp spin (libgomp.so.1 => /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1)
const big=23 small=4  region_us(big) med=6.9 p90=7.0  distinct_tids=23
alt   big=23 small=4  region_us(big) med=256.0 p90=287.2  region_us(small) med=6.5  distinct_tids=19004
== omp passive (libgomp.so.1 => /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1)
const big=23 small=4  region_us(big) med=23.2 p90=25.0  distinct_tids=23
alt   big=23 small=4  region_us(big) med=289.9 p90=314.2  region_us(small) med=18.1  distinct_tids=19004
== omp spin (libgomp.so.1 => /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1)
const big=23 small=4  region_us(big) med=6.8 p90=7.0  distinct_tids=23
alt   big=23 small=4  region_us(big) med=255.7 p90=290.1  region_us(small) med=6.5  distinct_tids=19004
```

## 2026-11-03T20:25:17.938Z

```
== gomp passive (libgomp.so.1 => /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1)
const big=23 small=4  region_us(big) med=36.9 p90=264.7  distinct_tids=23
alt   big=23 small=4  region_us(big) med=289.9 p90=316.0  region_us(small) med=17.8  distinct_tids=19004
const big=23 small=18  region_us(big) med=23.4 p90=25.3  distinct_tids=23
alt   big=23 small=18  region_us(big) med=69.6 p90=77.9  region_us(small) med=20.4  distinct_tids=5018
== gomp spin (libgomp.so.1 => /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1)
const big=23 small=4  region_us(big) med=6.9 p90=7.0  distinct_tids=23
alt   big=23 small=4  region_us(big) med=290.2 p90=315.2  region_us(small) med=6.7  distinct_tids=19004
const big=23 small=18  region_us(big) med=6.9 p90=7.1  distinct_tids=23
alt   big=23 small=18  region_us(big) med=53.2 p90=64.6  region_us(small) med=7.0  distinct_tids=5018
== omp passive (libgomp.so.1 => /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1)
const big=23 small=4  region_us(big) med=23.2 p90=25.2  distinct_tids=23
alt   big=23 small=4  region_us(big) med=287.1 p90=313.6  region_us(small) med=18.1  distinct_tids=19004
const big=23 small=18  region_us(big) med=23.2 p90=25.1  distinct_tids=23
alt   big=23 small=18  region_us(big) med=68.7 p90=76.3  region_us(small) med=20.3  distinct_tids=5018
== omp spin (libgomp.so.1 => /home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib/libgomp.so.1)
const big=23 small=4  region_us(big) med=6.9 p90=7.0  distinct_tids=23
alt   big=23 small=4  region_us(big) med=292.4 p90=313.5  region_us(small) med=6.7  distinct_tids=19004
const big=23 small=18  region_us(big) med=6.9 p90=7.0  distinct_tids=23
alt   big=23 small=18  region_us(big) med=54.9 p90=64.6  region_us(small) med=7.1  distinct_tids=5018
```

## 2026-11-03T20:25:27.872Z

```
0x000000000000000f (RPATH)              Library rpath: [/home/alacote/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/lib]
== libomp passive
   KMP_BLOCKTIME=0ms
   KMP_LIBRARY=throughput
   KMP_MWAIT_HINTS=0
const big=23 small=4  region_us(big) med=20.4 p90=38.3  distinct_tids=23
alt   big=23 small=4  region_us(big) med=18.9 p90=20.0  region_us(small) med=12.0  distinct_tids=23
const big=23 small=18  region_us(big) med=18.0 p90=19.9  distinct_tids=23
alt   big=23 small=18  region_us(big) med=19.4 p90=20.6  region_us(small) med=16.9  distinct_tids=23
== libomp spin
   KMP_BLOCKTIME=200ms
   OMP_WAIT_POLICY=active
   KMP_BLOCKTIME=200ms
const big=23 small=4  region_us(big) med=5.9 p90=6.0  distinct_tids=23
alt   big=23 small=4  region_us(big) med=12.6 p90=12.9  region_us(small) med=9.9  distinct_tids=23
const big=23 small=18  region_us(big) med=5.9 p90=6.0  distinct_tids=23
alt   big=23 small=18  region_us(big) med=9.9 p90=10.1  region_us(small) med=7.2  distinct_tids=23
```

