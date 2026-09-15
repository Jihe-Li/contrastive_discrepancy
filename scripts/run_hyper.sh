#!/usr/bin/env bash
# Testing-time hyperparameter selection: pick the number of Gaussian primitives
# per case by minimising CD, without using any landmark.
set -e

for case_idx in $(seq 1 10); do
  python hyperopti.py datasets.case_idx="${case_idx}"
done
