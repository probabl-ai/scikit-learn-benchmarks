#!/bin/bash
# Exp 26b: active cap of 16 against 20 between 20M and 50M. usage: run_exp26b.sh SLOT "MAXT|PROFILE|KEY=VAL" ...
export SLOT=$1; shift
export WORK=covtype,O:300000:100,h-25M:1000000:25,b-30M:600000:50,t-30M:1500000:20,b-36M:120000:300,h-40M:200000:200,h-40M-l127:200000:200:127,year_prediction_msd,b-45M:1500000:30,t-48M:800000:60,b-50M:250000:200,t-22M:110000:200
BASE=c325knob:main~ASTAIR=200000:4/5000000:8/20000000:16/999999999999
export VARS=$BASE:16,$BASE:20,$BASE:24
exec bash ~/work/hgb-regions/run_exp26.sh "$@"
