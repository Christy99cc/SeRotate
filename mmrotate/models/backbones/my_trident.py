# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class MyTrident(RotatedTwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MyTrident, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
