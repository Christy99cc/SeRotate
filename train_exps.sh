#!/usr/bin/env bash

#echo 'rotated_faster_rcnn_r50_fpnse31_1x_dota_le90_ss.py'
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/se/rotated_faster_rcnn_r50_fpnse31_1x_dota_le90_ss.py 4

#sleep 3s
#echo 'rotated_faster_rcnn_r50_fpnse30_1x_dota_le90_ss_02.py'
#CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/se/rotated_faster_rcnn_r50_fpnse30_1x_dota_le90_ss_02.py 2



# 训练
#sleep 3s
#CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse44s_1x_dota_le90.py 2
#sleep 2s
#CUDA_VISIBLE_DEVICES=0 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse44s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse44s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos44s1

#sleep 3s
#CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse55s_1x_dota_le90.py 2
#CUDA_VISIBLE_DEVICES=1 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse55s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse55s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos55s

#CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse48s_1x_dota_le90.py 2
#CUDA_VISIBLE_DEVICES=1 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse48s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse48s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos48s

#sleep 3s
#CUDA_VISIBLE_DEVICES=4,5 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse43s_1x_dota_le90.py 2
#CUDA_VISIBLE_DEVICES=5 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse43s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse43s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos43s_40

#sleep 3s
#CUDA_VISIBLE_DEVICES=4,5 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse48s_1x_dota_le90.py 2
#sleep 3s
#CUDA_VISIBLE_DEVICES=4,5 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse58s_1x_dota_le90.py 2
#CUDA_VISIBLE_DEVICES=5 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse58s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse58s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos58s_40_02
#CUDA_VISIBLE_DEVICES=5 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse48s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse48s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos48s_40_02



# 训练  80
sleep 3s
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/se/rotated_fcos_r50_fpnse42s_1x_dota_le90.py 2
sleep 2s
CUDA_VISIBLE_DEVICES=3 python ./tools/test.py configs/se/rotated_fcos_r50_fpnse42s_1x_dota_le90.py ./work_dirs/rotated_fcos_r50_fpnse42s_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos42s_80

