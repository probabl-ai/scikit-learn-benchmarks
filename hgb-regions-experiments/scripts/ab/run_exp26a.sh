#!/bin/bash
# Exp 26a: active caps of 16, 32 and 64 on all the shapes above 20M. usage: run_exp26a.sh SLOT "MAXT|PROFILE|KEY=VAL" ...
export SLOT=$1; shift
export WORK=covtype,O:300000:100,h-25M:1000000:25,b-30M:600000:50,b-36M:120000:300,h-40M:200000:200,h-40M-l127:200000:200:127,year_prediction_msd,b-45M:1500000:30,b-50M:250000:200,b-60M:600000:100,b-60M-t:4000000:15,b-60M-w:120000:500,b-60M-l127:400000:150:127,susy,b-75M:150000:500,h-80M:8000000:10,L-stumps,b-100M:1000000:100,b-120M:400000:300,h-200M:2000000:100
BASE=c325knob:main~ASTAIR=200000:4/5000000:8/20000000:16/999999999999
export VARS=$BASE:16,$BASE:32,$BASE:64
exec bash ~/work/hgb-regions/run_exp26.sh "$@"
