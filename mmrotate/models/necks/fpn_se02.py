import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class FPNSE02(nn.Module):
    """
    把FPNSE中卷积核大小由3, 7, 15改为 1, 3, 5
    FPNSE loss降不下来考虑是卷积核尺寸太大 e.g. 15x15

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
        super(FPNSE02, self).__init__()
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
                out_channels,
                3,
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

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level  # extra_levels = 1
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
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

        ## no use
        # 01. 横向连接与上采样结果进行cat
        # 中间cat结果
        # inter_results = [torch.cat([lateral, upsamples_results[i + 1]], dim=1) for i, lateral in enumerate(laterals)]

        # 02. 做fpnseconv
        # fpnse_conv_outs = [fpnse_conv(inter_results[i]) for i, fpnse_conv in enumerate(self.fpnse_convs)]
        ## no use end

        # build outputs
        # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]

        outs = []
        # print("used_backbone_levels", used_backbone_levels)
        for i in range(used_backbone_levels):
            out, self.kernel_out = self.fpnse_convs[i](laterals[i])
            channels = out.shape[1]

            out_fused = out[:, 0: self.out_channels, :, :]
            j = self.out_channels
            while j < channels:
                out_fused += out[:, j:j + self.out_channels, :, :]
                j = j + self.out_channels

            outs.append(out_fused)

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
        return tuple(outs), self.kernel_out


class FPNSE_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.w1 = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size, self.kernel_size),
                               requires_grad=True)
        self.w2 = None
        self.w3 = None

    def forward(self, inputs):

        w2_center_clip = self.w1[:, :, 1:-1, 1:-1]  # 3 x 3, [256, 256, 3, 3]
        w3_center_clip = self.w1[:, :, 2:-2, 2:-2]  # 1 x 1, [256, 256, 1, 1]

        # print("w2_center_clip.shape:", w2_center_clip.shape)
        # print("w3_center_clip.shape:", w3_center_clip.shape)

        size = [self.kernel_size, self.kernel_size]
        self.w2 = F.interpolate(w2_center_clip, mode="nearest", size=size) / 2e3 # [256, 256, 5, 5]
        self.w3 = F.interpolate(w3_center_clip, mode="nearest", size=size) / 2e5 # [256, 256, 5, 5]


        weight = torch.cat([self.w1, self.w2, self.w3], 0)  # [, , 15, 15]
        outputs = F.conv2d(inputs, weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return outputs, weight
