import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init, build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch.nn.modules.utils import _pair
from torch.nn import init

from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class SKNetNeck(nn.Module):
    """
    分支注意力
    use SKNet的思想

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
        super(SKNetNeck, self).__init__()
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
        self.conv1x1s = nn.ModuleList()

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
            # fpnse_conv = AtrousSE(out_channels, out_channels)
            fpnse_conv = SKConv(out_channels, WH=1, M=2, G=1, r=2)

            self.fpnse_convs.append(fpnse_conv)

            conv1x1 = ConvModule(
                # 3 * out_channels,
                out_channels,
                out_channels,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.conv1x1s.append(conv1x1)

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

        outs = []
        # print("used_backbone_levels", used_backbone_levels)
        for i in range(used_backbone_levels):
            # if i > 0:
            #     outs.append(self.fpn_convs[i](laterals[i]))
            #     continue

            out = self.fpnse_convs[i](laterals[i])
            out = self.conv1x1s[i](out)
            outs.append(out)

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = outs[i - 1].shape[2:]
            # upsamples_results[i] = F.interpolate(
            #     laterals[i], size=prev_shape, **self.upsample_cfg)
            outs[i - 1] += F.interpolate(
                outs[i], size=prev_shape, **self.upsample_cfg)


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
                 M=2, L= 32, r=2, # attention params
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

        # attention
        self.M = len(self.dilations)
        d = max(int(self.M * out_channels / r), L)
        self.fc = nn.Linear(self.M * out_channels, d)
        self.fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(nn.Linear(d, out_channels))
        self.softmax = nn.Softmax(dim=1)


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
        ]  # [2, 256, 256, 256]

        feas = torch.stack(outputs).transpose(0, 1)  # [ 2, N, 256,...]
        fea_U = feas.reshape(feas.shape[0], feas.shape[1] * feas.shape[2], feas.shape[3], feas.shape[4])  # [2, 3 * 256, ...]
        fea_s = fea_U.mean(-1).mean(-1)  # [2, 3*256]
        fea_z = self.fc(fea_s) # [2, 3*128]  # [2, 384]
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector  # [2,1,256]
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)   # [2, 3, 256]
        attention_vectors = self.softmax(attention_vectors)  # [2, 3, 256]
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)  # [2, M, 256, 1, 1]
        # + ##
        # fea_v = (feas * attention_vectors).sum(dim=1)
        #
        # cat ##
        fea_v = feas * attention_vectors
        fea_v = fea_v.reshape(fea_v.shape[0], fea_v.shape[1] * fea_v.shape[2], fea_v.shape[3], fea_v.shape[4])
        #
        return fea_v


class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            # print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v



