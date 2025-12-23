#!/bin/bash       

python experiment_runner_v5.py \
  --stage 1 \
  --output_dir "experiments_final" \
  --seed 42

python experiment_runner_v5.py \
  --stage 2 \
  --output_dir "experiments_final" \
  --seed 42

python experiment_runner_v5.py \
  --stage 3 \
  --output_dir "experiments_final" \
  --seed 42

python experiment_runner_v5.py \
  --stage 4 \
  --output_dir "experiments_final" \
  --seed 42

python experiment_runner.py \
  --stage 1 \
  --output_dir "experiments_final_2" \
  --seed 42

python experiment_runner.py \
  --stage 2 \
  --output_dir "experiments_final_2" \
  --seed 42

python experiment_runner.py \
  --stage 3 \
  --output_dir "experiments_final_2" \
  --seed 42

python experiment_runner.py \
  --stage 4 \
  --output_dir "experiments_final_2" \
  --seed 42

mv -r experients_final_2/stage4_color_temperature experiments_final
rm -r experients_final_2