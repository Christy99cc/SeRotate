#!/usr/bin/bash

OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/faster_myneck8_1x.py 1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/faster_myneck8_1x.py ./work_dirs/faster_myneck8_1x/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_faster_myneck8_1x_1119



