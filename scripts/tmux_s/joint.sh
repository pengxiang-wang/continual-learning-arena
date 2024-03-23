#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {4..10}
do
    python src/train.py trainer=gpu trainer.devices=[1] experiment=Joint_1 seed=$i
done