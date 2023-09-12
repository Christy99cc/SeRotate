#python ./tools/test.py \
#  configs/se/rotated_fcos_r50_fpnse42s_1x_dota_le90.py \
#  /home/zsx/workzsx/projects/mmrotate/fpnse10/work_dirs/rotated_fcos_r50_fpnse42s_1x_dota_le90/epoch_12.pth \
#  --show-dir work_dirs/MyTest-vis-se42
#
#python ./tools/test.py \
#  configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py \
#  /home/zsx/workzsx/projects/mmrotate/fpnse/work_dirs/rotated_fcos_r50_fpn_1x_dota_le90/epoch_12.pth \
#  --show-dir work_dirs/MyTest-vis-bs

python demo/huge_image_demo.py \
    demo/dota_demo.jpg \
    configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_v3.py \
    checkpoint/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
