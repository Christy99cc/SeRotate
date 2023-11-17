# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import (RotatedConvFCBBoxHead,
                                RotatedKFIoUShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead_CosLoss1)
from .gv_bbox_head import GVBBoxHead
from .rotated_bbox_head import RotatedBBoxHead


__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'GVBBoxHead', 'RotatedKFIoUShared2FCBBoxHead',
    'RotatedShared2FCBBoxHead_CosLoss1'
]
