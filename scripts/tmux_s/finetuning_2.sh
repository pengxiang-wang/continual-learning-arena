#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..5}
do
    python src/train.py experiment=Finetuning_2 trainer=gpu trainer.devices=[0] seed=$i
done