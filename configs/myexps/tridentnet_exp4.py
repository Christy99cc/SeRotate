_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py', '../_base_/models/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
]
angle_version = 'le90'

model = dict(
    type='MyTridentFasterRCNN',
    backbone=dict(
        type='MyTridentResNet',
        trident_dilations=(1, 2, 3),
        strides=(1, 2, 2),
        dilations=(1, 2, 3),
        frozen_stages=1,
        num_stages=3,
        out_indices=(2,),
        num_branch=3,
        test_branch_idx=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=None,
    rpn_head=dict(
        type='RotatedRPNHead',
        in_channels=1024,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            # scales=[8],
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            # strides=[4, 8, 16, 32, 64]
            strides=[16]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='MyTridentRoIHead', num_branch=3, test_branch_idx=1,
        version=angle_version,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            # featmap_strides=[4, 8, 16, 32]
            featmap_strides=[16]
        ),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            # in_channels=256,
            in_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHAHBBoxCoder',
                angle_range=angle_version,
                norm_factor=2,
                edge_swap=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),

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
evaluation = dict(interval=2, metric='mAP')

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# dataset settings
dataset_type = 'DOTADataset'
# data_root = 'data/split_ss_dota1_0/'
data_root = 'data/split_ss_1024_dota/'
# data_root = 'data/split_ms_dota/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))


# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=0 ./tools/dist_train.sh configs/myexps/tridentnet_exp4.py 4
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=0 python ./tools/test.py configs/myexps/tridentnet_exp4.py ./work_dirs/tridentnet_exp4/epoch_12.pth --format-only --eval-options submission_dir=work_dirs/Task1_results_tridentnet_exp4_1024
