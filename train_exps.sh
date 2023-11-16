#!/usr/bin/bash
#sleep 2s
OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/fcos_myneck5_1x.py 1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/fcos_myneck5_1x.py ./work_dirs/fcos_myneck5_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_fcos_myneck5_1x_1110
