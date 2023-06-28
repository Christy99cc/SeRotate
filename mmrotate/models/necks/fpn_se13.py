import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from torch.nn import init

from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class FPNSE13(nn.Module):
    """
    将fpn中的3x3改为5x5，输出维度改为768，forward里分成3块相加再除以3
    自己实现上述conv模块
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
        super(FPNSE13, self).__init__()
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
        self.fpn_convs = nn.ModuleList()
        self.fpnse_convs = nn.ModuleList()
        kernels_size = [15, 7, 3]

        for i in range(self.start_level, self.backbone_end_level):  # range(0, 4)
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                3 * out_channels,
                5,  ## 5x5
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            # if i < 3:
            fpnse_conv = FPNSE_Conv(out_channels, out_channels)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpnse_convs.append(fpnse_conv)

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
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        outs = []
        # print("used_backbone_levels", used_backbone_levels)
        for i in range(used_backbone_levels):
            out, _ = self.fpnse_convs[i](laterals[i])  ## 自己实现的
            # print("=====", out.shape)
            channels = out.shape[1]

            out_fused = out[:, 0: self.out_channels, :, :]
            j = self.out_channels
            while j < channels:
                out_fused += out[:, j:j + self.out_channels, :, :]
                j = j + self.out_channels

            out_fused = out_fused / 3.0
            outs.append(out_fused)

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs), self.kernel_out


class FPNSE_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.w = nn.Parameter(torch.randn(3 * out_channels, in_channels, self.kernel_size, self.kernel_size),
                              requires_grad=True)
        self.bias = nn.Parameter(torch.empty(3 * out_channels), requires_grad=True)

        self.reset_parameters()

    # from pytorch
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
    def forward(self, inputs):
        outputs = F.conv2d(inputs, self.w, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return outputs, self.w
