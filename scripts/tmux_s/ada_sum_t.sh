#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {3..10}
do
    python src/train.py trainer=gpu experiment=AdaHAT_1 experiment_name=ada_sum_t model.adjust_strategy=ada_sum_t seed=$i
done