#!/bin/bash
#$ -q meta_core.q
#$ -N mllab-test-rnd-10-precision
#$ -m ea
#$ -v PYTHONPATH=:/home/arabim/mllab/venv/bin
#$ -M arabim@informatik.uni-freiburg.de
timeout 5000 /home/arabim/mllab/venv/bin/python /home/arabim/mllab/workstation.py calc -i 10 --raw -s raw-test-rnd-10-precision.json -t 1 2 3 4 5 6 7 9 10 11 12 13 14 15 16 18 20 21 22 23 24 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 45 47 48 49 50 51 52 53 54 55 57 58 59 60 145 206 219 2065 2067 2068 2071 2073 2074 2075 2076 2077 2078 2079 2142 2146 2272 2273 2274 2275 2276 2372 2373 2382 3010 3011 3012 3018 3019 3021 3022 3481 3483 3484 3485 3486 3487 3488 3491 3492 3493 3494 3495 3496 3497 3498 3499
