#!/usr/bin/bash
#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/oriented_rcnn_myneck9_1x.py 1
#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/oriented_rcnn_myneck9_1x.py ./work_dirs/oriented_rcnn_myneck9_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_oriented_rcnn_myneck9_1x_1120

#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 1
#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_faster_rcnn_r50_fpn_1x_dota_le90_1121

#OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck8_1x.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck8_1x.py ./work_dirs/faster_myneck8_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck8_1x_1123
#OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck11_1x.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck11_1x.py ./work_dirs/faster_myneck11_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck11_1x_1124

#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/oriented_rcnn_myneck8_1x.py 1
#CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/oriented_rcnn_myneck8_1x.py ./work_dirs/oriented_rcnn_myneck8_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_oriented_rcnn_myneck8_1x_1122
#
#OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck12_1x.py 1
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck12_1x.py ./work_dirs/faster_myneck12_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck12_1x_1124


OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck13_1x.py 1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck13_1x.py ./work_dirs/faster_myneck13_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck13_1x_1128
