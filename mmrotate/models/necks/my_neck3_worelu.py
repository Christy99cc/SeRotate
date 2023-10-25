import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init, build_activation_layer, build_norm_layer
from torch.nn import init

from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class MyNeck3(nn.Module):
    """
    等变卷积核大小 5 x 5
    1 3 5

    w1->w2：clip + 填补
    w2->w3: clip + 填补



    relu
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
        super(MyNeck3, self).__init__()
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

        for i in range(self.start_level, self.backbone_end_level):
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

            # if i == 1:
            fpnse_conv = FPNSE_Conv(out_channels, out_channels)
            self.fpnse_convs.append(fpnse_conv)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

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

        outs = []
        # print("used_backbone_levels", used_backbone_levels)
        for i in range(used_backbone_levels):
            # if i > 0:
            #     outs.append(self.fpn_convs[i](laterals[i]))
            #     continue

            out, self.kernel_out = self.fpnse_convs[i](laterals[i])
            channels = out.shape[1]

            out_fused = out[:, 0: self.out_channels, :, :]
            j = self.out_channels
            while j < channels:
                out_fused = out_fused + out[:, j:j + self.out_channels, :, :]
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


class FPNSE_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0, dilation=1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.w1 = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size, self.kernel_size),
                               requires_grad=True)
        self.w2_t = nn.Parameter(
            torch.randn(out_channels, in_channels, self.kernel_size ** 2 - (self.kernel_size // 2 + 1) ** 2),
            requires_grad=True)
        self.w3_t = nn.Parameter(
            torch.randn(out_channels, in_channels, self.kernel_size ** 2 - (self.kernel_size // 2 + 1) ** 2),
            requires_grad=True)
        self.bias_t = nn.Parameter(torch.empty(out_channels), requires_grad=True)

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
        init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        init.kaiming_uniform_(self.w2_t, a=math.sqrt(5))
        init.kaiming_uniform_(self.w3_t, a=math.sqrt(5))

        if self.bias_t is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w1)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_t, -bound, bound)

    def forward(self, inputs):
        # print(self.w1.shape, self.w2.shape, self.w3.shape)
        out_channels = self.out_channels
        in_channels = self.in_channels

        # init w2
        self.w2 = []
        k, l = 0, 0
        w1_clip = self.w1[:, :, 1:-1, 1:-1].reshape(out_channels, in_channels, -1)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if j % 2 == 0 and i % 2 == 0:
                    self.w2.append(w1_clip[:, :, l])
                    l += 1
                else:
                    self.w2.append(self.w2_t[:, :, k])
                    k += 1
        self.w2 = torch.cat(self.w2, dim=-1).reshape(out_channels, in_channels, self.kernel_size, self.kernel_size)

        # init w3
        self.w3 = []
        k, l = 0, 0
        w2_clip = self.w2[:, :, 1:-1, 1:-1].reshape(out_channels, in_channels, -1)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if j % 2 == 0 and i % 2 == 0:
                    self.w3.append(w2_clip[:, :, l])
                    l += 1
                else:
                    self.w3.append(self.w3_t[:, :, k])
                    k += 1
        self.w3 = torch.cat(self.w3, dim=-1).reshape(out_channels, in_channels, self.kernel_size, self.kernel_size)

        self.w = torch.cat([self.w1, self.w2, self.w3], dim=0)  # [3*out_channel, in_channel, kernel_size, kernel_size]
        self.bias = torch.cat([self.bias_t, self.bias_t, self.bias_t], dim=0)

        outputs = F.conv2d(inputs, self.w, bias=self.bias, stride=self.stride, padding=self.padding,
                           dilation=self.dilation)

        outputs = F.normalize(outputs)
        o = F.relu(outputs)

        return o, self.w


if __name__ == "__main__":
    net = FPNSE_Conv(32, 32, 5, 1)
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
