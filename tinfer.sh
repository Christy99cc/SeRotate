#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_myneck4_3x_exp1.py 4
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/rotated_fcos_r50_myneck4_3x_exp1.py ./work_dirs/rotated_fcos_r50_myneck4_3x_exp1/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_rotated_fcos_r50_myneck4_3x_exp1_1026
