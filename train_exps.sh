#!/usr/bin/env bash
#sleep 2s
#CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_myfpnse48s_1x_dota_le90.py 1
#CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_myfpnse48s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_myfpnse48s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_myfpnse48s_1x_dota_le90_0927

#CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_myneck2_1x_dota_le90.py 2

#CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_fpn_1x_dota_le90.py 2

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_myneck3_1x_dota_le90.py 2

#CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_myneck2_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_myneck2_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_myneck2_1x_dota_le90_1024

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/dev.py 1
#CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_myneck2_1x_dota_le90.py 2

#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_myneck2_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_myneck2_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_myneck2_1x_dota_le90_1024
#CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_myneck3_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_myneck3_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_myneck3_1x_dota_le90_1025

# rotated_fcos_r50_myneck3_1x_dota_le90-exp2.py

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/tridentnet_exp3.py 4
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/tridentnet_exp3.py ./work_dirs/tridentnet_exp3/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_tridentnet_exp3_1024
