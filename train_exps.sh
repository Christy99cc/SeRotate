#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/oriented_rcnn_myneck9_1x.py 1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/oriented_rcnn_myneck9_1x.py ./work_dirs/oriented_rcnn_myneck9_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_oriented_rcnn_myneck9_1x_1120


