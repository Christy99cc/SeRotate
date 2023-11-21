import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init, build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch.nn.modules.utils import _pair
from torch.nn import init

from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class MyNeck10(nn.Module):
    """
    去掉short-connection

    只用一层dilation + 2个shared3x3

    适用于fasterrcnn
    暂不适用于RetinaNet

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(MyNeck10, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.kernel_out = []

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        # self.conv1x1s = nn.ModuleList()
        # self.se_conv2s = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

            # conv1x1 = ConvModule(
            #     in_channels[i] + out_channels,
            #     out_channels,
            #     1,
            #     padding=0,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg,
            #     act_cfg=act_cfg,
            #     inplace=False)
            #
            # self.conv1x1s.append(conv1x1)

            # if i < 2:
            #     if i == 1:
            #         se_conv2 = AtrousSEV2(in_channels=out_channels, out_channels=out_channels,
            #                               strides=(1, 2, 2, 4), dilations=(1, 2, 4, 8))
            #     if i == 2:
            #         se_conv2 = AtrousSEV2(in_channels=out_channels, out_channels=out_channels,
            #                               strides=(1, 1, 2, 2), dilations=(1, 2, 4, 8))
            #     self.se_conv2s.append(se_conv2)

        self.fpn_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

        self.se_conv = AtrousSE(in_channels=out_channels, out_channels=out_channels,
                                strides=(1, 1, 1, 1), dilations=(1, 2, 4, 8))

        # add extra conv layers (e.g., RetinaNet)
        # extra_levels = num_outs - self.backbone_end_level + self.start_level  # extra_levels = 1
        # if self.add_extra_convs and extra_levels >= 1:
        #     for i in range(extra_levels):
        #         if i == 0 and self.add_extra_convs == 'on_input':
        #             in_channels = self.in_channels[self.backbone_end_level - 1]
        #         else:
        #             in_channels = out_channels
        #         extra_fpn_conv = ConvModule(
        #             in_channels,
        #             out_channels,
        #             3,
        #             stride=2,
        #             padding=1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #             inplace=False)
        #         self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function"""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # upsamples_results = [0] * 4
        # build top-down path
        used_backbone_levels = len(laterals)  # 4

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # upsamples_results[i] = F.interpolate(
            #     laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        # P2 作为融合的特征
        # 对P2做尺度等变，然后再加回原PFN出的结果上
        p2 = self.fpn_conv(laterals[0])
        se_outs = self.se_conv(p2)
        # len(outs) = 4
        # torch.Size([2, 256, 256, 256])
        # torch.Size([2, 256, 128, 128])
        # torch.Size([2, 256, 64, 64])
        # torch.Size([2, 256, 32, 32])

        # cat Ci and outs, i = 2, 3, 4, 5
        # cat_outs = [torch.cat([se_outs[i], inputs[i + self.start_level]], dim=1)
        #             for i in range(len(se_outs))]
        #
        # outs_1x1s = [self.conv1x1s[i](cat_outs[i]) for i in range(len(cat_outs))]

        outs = se_outs

        # # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)


class AtrousSE(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=(1, 1, 1), dilations=(1, 2, 3),
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.kernel_size = _pair(kernel_size)
        self.paddings = _pair(dilations)
        self.dilations = dilations

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    # from pytorch
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        outputs = [
            F.conv2d(inputs, self.weight, self.bias, stride, padding,
                     dilation) for stride, dilation, padding in zip(
                self.strides, self.dilations, self.paddings)
        ]
        return outputs


class AtrousSEV2(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=(1, 1, 1), dilations=(1, 2, 3),
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.kernel_size = _pair(kernel_size)
        self.paddings = _pair(dilations)
        self.dilations = dilations

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    # from pytorch
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: List):
        outputs = [
            F.conv2d(input, self.weight, self.bias, stride, padding,
                     dilation) for input, stride, dilation, padding in zip(
                inputs, self.strides, self.dilations, self.paddings)
        ]
        return outputs


if __name__ == "__main__":
    net = AtrousSE(32, 32, 3, 1)
    # w = net.w
    # w1 = w[0:32]
    # w2 = w[32:64]
    # w3 = w[64:]
    # print(w1[0,0,1,1], w2[0,0,0,0])
    # print(w2[0,0,1,1], w3[0,0,0,0])

    inp = torch.randn([2, 32, 128, 128])
    ref = torch.ones([2, 96, 124, 124])
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    for iter in range(100):
        opt.zero_grad()
        out, w = net(inp)
        # print(out.shape)
        loss = ((ref - out) ** 2).mean()
        loss.backward()
        opt.step()
        print(loss.item())
        w1 = w[0:32]
        w2 = w[32:64]
        w3 = w[64:]
        print(w1[0, 0, 1, 1], w2[0, 0, 0, 0])
        print(w2[0, 0, 1, 1], w3[0, 0, 0, 0])
