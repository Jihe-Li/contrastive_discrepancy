#!/usr/bin/env bash
# Sweep model complexity over the full DIRLab dataset and record CD next to the
# reference TRE. This reproduces the dataset-level curves of the paper.
set -e

for case_idx in $(seq 1 10); do
  for num_gaussians in 200 400 800 1600 3200 6400 12800 25600 51200 102400 204800; do
    python run.py \
      datasets.case_idx="${case_idx}" \
      network.num_gaussians="${num_gaussians}" \
      metric=both
  done
done
