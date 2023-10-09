_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py', '../_base_/models/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
]
angle_version = 'le90'

model = dict(
    type='RotatedFasterRCNN',
    backbone=dict(
        type='MyTridentResNet',
        trident_dilations=(1, 2, 3),
        strides=(1, 2, 2),
        dilations=(1, 2, 3),
        num_stages=3,
        out_indices=(0, 1, 2),
        num_branch=3,
        test_branch_idx=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    roi_head=dict(type='MyTridentRoIHead', num_branch=3, test_branch_idx=1),
    train_cfg=dict(
        rpn_proposal=dict(nms_post=500, max_num=500),
        # rcnn=dict(
        #     sampler=dict(num=128, pos_fraction=0.5,
        #                  add_gt_as_proposals=False))
    ))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

optimizer = dict(lr=0.005)
evaluation = dict(interval=1, metric='mAP')

# configs/myexps/tridentnet_exp1.py
# CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/tridentnet_exp1.py 2
# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/tridentnet_exp1.py 1
