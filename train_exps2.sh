#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpn_1x_hrsc_le90.py 2
#


# 训练
#sleep 3s
#CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse46s_1x_dota_le90.py 2
#CUDA_VISIBLE_DEVICES=1 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse46s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse46s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos46s
#
#sleep 3s
#CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse46s_1x_dota_le90.py 2
#CUDA_VISIBLE_DEVICES=1 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse46s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse46s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos46s


## 训练
#sleep 3s
#echo 'rotated_faster_rcnn_r50_fpnse45_1x_dota_le90_ss.py'
#CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/se/rotated_faster_rcnn_r50_fpnse45_1x_dota_le90_ss.py 2
#sleep 3s
#echo 'rotated_faster_rcnn_r50_fpnse46_1x_dota_le90_ss.py'
#CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/se/rotated_faster_rcnn_r50_fpnse46_1x_dota_le90_ss.py 2
#sleep 3s
#echo 'rotated_faster_rcnn_r50_fpnse47_1x_dota_le90_ss.py'
#CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/se/rotated_faster_rcnn_r50_fpnse47_1x_dota_le90_ss.py 2
#
## 测试
#sleep 3s
#echo 'test start'
#CUDA_VISIBLE_DEVICES=1 python ./tools/test.py configs/se/rotated_faster_rcnn_r50_fpnse45_1x_dota_le90_ss.py ./work_dirs/rotated_faster_rcnn_r50_fpnse45_1x_dota_le90_ss/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results45
#CUDA_VISIBLE_DEVICES=0 python ./tools/test.py configs/se/rotated_faster_rcnn_r50_fpnse46_1x_dota_le90_ss.py ./work_dirs/rotated_faster_rcnn_r50_fpnse46_1x_dota_le90_ss/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results46
#CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/se/rotated_faster_rcnn_r50_fpnse47_1x_dota_le90_ss.py ./work_dirs/rotated_faster_rcnn_r50_fpnse47_1x_dota_le90_ss/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results47
#

CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_atrous_1x_dota_le90.py 2
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_atrous_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_atrous_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_atrous_1x_dota_le90_0915
