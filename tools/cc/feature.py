# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed

from mmrotate.apis import train_detector
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import (collect_env, get_device, get_root_logger,
                            setup_multi_processes)
from mmrotate.models.necks import fpn_se48_s


def main():
    img = 'xxx'
    covnet = fpn_se48_s.FPNSE_Conv(in_channels=256, out_channels=256)
    feature, w12 = covnet(img)
#   plot feature



if __name__ == '__main__':
    main()
