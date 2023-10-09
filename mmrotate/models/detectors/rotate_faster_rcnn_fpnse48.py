# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector
from torch import nn
from torch.nn import functional as F



@ROTATED_DETECTORS.register_module()
class RotatedFasterRCNNFPNSE48(RotatedTwoStageDetector):
    """Implementation of Rotated `Faster R-CNN.`__

    __ https://arxiv.org/abs/1506.01497
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedFasterRCNNFPNSE48, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # self.kernel_out = []
        # self.mse_loss = nn.MSELoss()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x, _ = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # x = self.extract_feat(img)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def _kernel_loss(self):
        kernel_out = self.kernel_out
        w1 = kernel_out[0]  # 15x15 =>
        w2 = kernel_out[1]  # 15x15 => 7x7
        w3 = kernel_out[2]  # 15x15 => 3x3

        print("w1.shape:", w1.shape)
        print("w2.shape:", w2.shape)
        print("w3.shape:", w3.shape)

        def max_pooling(input, scale=2):
            dim = input.shape  # [bs， c, w, h]
            return F.max_pool2d(input, kernel_size=scale) \
                .view(-1, dim[1], dim[2] // scale, dim[3] // scale)

        w2_center_clip = w2[:, :, 4:-4, 4:-4]  # 7 x 7, [256, 64, 7, 7]
        w3_center_clip = w3[:, :, 6:-6, 6:-6]  # 3 x 3, [256, 64, 3, 3]
        print("w2_center_clip.shape:", w2_center_clip.shape)
        print("w3_center_clip.shape:", w3_center_clip.shape)

        max_pooling_w1 = max_pooling(w1)
        max_pooling_w2 = max_pooling(w2, scale=4)
        print("max_pooling_w1.shape:", max_pooling_w1.shape)
        print("max_pooling_w2.shape:", max_pooling_w2.shape)
        loss = self.mse_loss(w2_center_clip, max_pooling_w1).mean() + self.mse_loss(w3_center_clip,
                                                                                    max_pooling_w2).mean()
        return loss


