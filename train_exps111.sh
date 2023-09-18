#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/rotated_fcos_r50_atrous_1x_dota_le90111.py 2
