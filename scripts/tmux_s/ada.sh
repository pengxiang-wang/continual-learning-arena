#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..10}
do
    python src/train.py trainer=gpu experiment=AdaHAT_1 experiment_name=ada model.adjust_strategy=ada seed=$i
done