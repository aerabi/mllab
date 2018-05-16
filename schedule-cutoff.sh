#!/bin/bash
#$ -q meta_core.q
#$ -N mllab-cutoff-50
#$ -m ea
#$ -v PYTHONPATH=:/home/arabim/mllab/venv/bin
#$ -M arabim@informatik.uni-freiburg.de
timeout 5000 /home/arabim/mllab/venv/bin/python /home/arabim/mllab/find_cut_off.py calc -p /home/arabim/mllab/venv/bin -w /home/arabim/mllab -s 1 2 4 8 16 32 64 128 256 512 1024 -i 1 -S cutoff50.json -c 50
