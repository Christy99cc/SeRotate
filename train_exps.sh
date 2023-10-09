#!/usr/bin/env bash
#sleep 2s
#CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_myfpnse48s_1x_dota_le90.py 1
#CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_myfpnse48s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_myfpnse48s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_myfpnse48s_1x_dota_le90_0927

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/tridentnet_exp1.py 1
