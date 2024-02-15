##!/usr/bin/bash
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/oriented_rcnn_myneck9_1x.py 1
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/oriented_rcnn_myneck9_1x.py ./work_dirs/oriented_rcnn_myneck9_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_oriented_rcnn_myneck9_1x_1120
#
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 1
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_faster_rcnn_r50_fpn_1x_dota_le90_1209
#
##OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck8_1x.py 1
##CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck8_1x.py ./work_dirs/faster_myneck8_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck8_1x_1123
##OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck11_1x.py 1
##CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck11_1x.py ./work_dirs/faster_myneck11_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck11_1x_1124
#
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/oriented_rcnn_myneck8_1x.py 1
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/oriented_rcnn_myneck8_1x.py ./work_dirs/oriented_rcnn_myneck8_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_oriented_rcnn_myneck8_1x_1122
##
##OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck12_1x.py 1
##CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck12_1x.py ./work_dirs/faster_myneck12_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck12_1x_1124
#
##
##OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck18_1x.py 1
##CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck18_1x.py ./work_dirs/faster_myneck18_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck18_1x_1208
#
#
##OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/oriented_rcnn_myneck17_1x.py 1
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/oriented_rcnn_myneck17_1x.py ./work_dirs/oriented_rcnn_myneck17_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_oriented_rcnn_myneck17_1x_1130
#
##OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_retinanet_obb_r50_myneck17_2_1x_dota_le90.py 1
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_retinanet_obb_r50_myneck17_2_1x_dota_le90.py ./work_dirs/rotated_retinanet_obb_r50_myneck17_2_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_retinanet_obb_r50_myneck17_1x_dota_le90_1202
#
##OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/oriented_reppoints_r50_myneck17_1x_dota_le135.py 1
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/oriented_reppoints_r50_myneck17_1x_dota_le135.py ./work_dirs/oriented_reppoints_r50_myneck17_1x_dota_le135/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_oriented_reppoints_r50_myneck17_1x_dota_le135_1201
##
#
##CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_faster_rcnn_r50_fpn_1x_dota_le90
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_hrsc_3x_02.py 1
#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck17_hrsc_3x_02.py 1
#
#CUDA_VISIBLE_DEVICES=5 python ./tools/test.py configs/myexps/faster_hrsc_3x.py /home/zsx/projects/mmrotate/faster_hrsc/work_dirs/faster_hrsc_3x/epoch_36.pth --eval mAP
#
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/s2anet/s2anet_r50_fpn_3x_hrsc_le135.py 1


#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_sknet_1x.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_sknet_1x.py ./work_dirs/faster_sknet_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_sknet_1x_0103

#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck17_1x.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck17_1x.py ./work_dirs/faster_myneck17_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck17_1x_0103


#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck19_relu_1x.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck19_relu_1x.py ./work_dirs/faster_myneck19_relu_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck19_relu_0203


CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck15_hrsc_3x.py 1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck15_hrsc_3x.py ./work_dirs/faster_myneck15_hrsc_3x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck15_hrsc_3x_0216
