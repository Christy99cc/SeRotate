#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse58s_1x_dota_le90.py 2
CUDA_VISIBLE_DEVICES=2 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse58s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse58s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos58s
