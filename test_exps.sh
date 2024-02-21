#CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=0 python ./tools/test.py \
#  configs/myexps/faster_myneck17_1x.py \
#  ./work_dirs/faster_myneck17_epoch_12.pth \
#  --show-dir work_dirs/vis-faster_myneck17_1x

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=0 python ./tools/test.py \
  configs/myexps/faster_myneck17_hrsc_3x_02.py \
  ./work_dirs/faster_myneck17_hrsc_epoch_12.pth \
  --show-dir work_dirs/vis-hrsc_faster_myneck17_1x