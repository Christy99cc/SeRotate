#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py 1
#CUDA_VISIBLE_DEVICES=0 python ./tools/test.py configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpn_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_fpn_1x_dota_le90_batch2
#CUDA_VISIBLE_DEVICES=0 python ./tools/test.py configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpn_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_fpn_1x_dota_le90_batch2


#
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse42s_1x_dota_le90.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse42s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse42s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_fpnse42s_1x_dota_le90_batch2

#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse44s_1x_dota_le90.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse44s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse44s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_fpnse44s_1x_dota_le90_batch2

#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_atrous_1x_dota_le90.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_atrous_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_atrous_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_atrous_1x_dota_le90_0913

#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_atrous_3x_dota_le90.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_atrous_3x_dota_le90.py ./work_dirs/rotated_fcos_r50_atrous_3x_dota_le90/epoch_36.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_atrous_3x_dota_le90_0913

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/tridentnet_exp1.py 2
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpn_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_fpn_1x_dota_le90_0926
