# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
# copyright (c) 2024 PytorchOCR Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PP-LCNet v1 backbone implementation for PyTorch.

PP-LCNet: A Lightweight CPU Convolutional Neural Network
Paper: https://arxiv.org/abs/2109.15099

This backbone implements the original PP-LCNet (v1) architecture used in
PaddleOCR's textline_ori and doc_ori models.
The backbone includes the GAP + 1280-dim 1x1 conv + H-Swish + Dropout
that is characteristic of PP-LCNet, allowing ClsHead to work directly.
"""

from __future__ import absolute_import, division, print_function

import torch.nn as nn


# PP-LCNet network configuration (original ImageNet style)
# Format: blocks{stage}: [kernel_size, in_channels, out_channels, stride, use_se]
NET_CONFIG = {
    "blocks2": [
        [3, 16, 32, 1, False],
    ],
    "blocks3": [
        [3, 32, 64, 2, False],
        [3, 64, 64, 1, False],
    ],
    "blocks4": [
        [3, 64, 128, 2, False],
        [3, 128, 128, 1, False],
    ],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [
        [5, 256, 512, 2, True],
        [5, 512, 512, 1, True],
    ],
}


def make_divisible(v, divisor=8, min_value=None):
    """Make channel divisible by divisor (same as PaddleClas)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DepthSepConv(nn.Module):
    """
    Depthwise Separable Convolution block (MobileNetV1 style).
    PP-LCNet uses simple DepthSepConv without residual connections.

    DW Conv(kxk, groups=in_ch) → BN → H-Swish → PW Conv(1x1) → BN → H-Swish
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.Hardswish(inplace=True)

        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_act(x)
        return x


class SELayer(nn.Module):
    """Squeeze-and-Excitation module (PP-LCNet uses reduction=4)."""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return identity * x


class PPLCNetBlock(nn.Module):
    """PP-LCNet basic block: DW Conv → BN → H-Swish → (SE) → PW Conv → BN → H-Swish."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_se=False):
        super().__init__()
        self.use_se = use_se
        # Depthwise conv
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.Hardswish(inplace=True)

        # SE on intermediate features (after DW, before PW)
        if use_se:
            self.se = SELayer(in_channels, reduction=4)

        # Pointwise conv
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_act(x)
        return x


class PPLCNet(nn.Module):
    """
    PP-LCNet v1 backbone for classification tasks.

    Used in PaddleOCR's textline_ori and doc_ori models.
    Outputs a 1280-dim feature vector (after GAP + 1x1 Conv + H-Swish + Dropout),
    suitable for direct use with ClsHead.

    Args:
        scale (float): Channel scaling factor (0.25, 0.35, 0.5, 0.75, 1.0, etc.)
        class_num (int): Number of output classes (NOT used here; ClsHead handles classification).
                         Included for compatibility but ignored.
    """

    def __init__(self, scale=1.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.net_config = NET_CONFIG

        # Stem: Conv2D(3→16*scale, k=3, s=2)
        stem_out = make_divisible(16 * scale)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=stem_out,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv1_bn = nn.BatchNorm2d(stem_out)
        self.conv1_act = nn.Hardswish(inplace=True)

        # Blocks 2-6 (stages)
        self.blocks2 = self._make_stage(self.net_config["blocks2"])
        self.blocks3 = self._make_stage(self.net_config["blocks3"])
        self.blocks4 = self._make_stage(self.net_config["blocks4"])
        self.blocks5 = self._make_stage(self.net_config["blocks5"])
        self.blocks6 = self._make_stage(self.net_config["blocks6"])

        # Last stage output channels (after scale)
        last_out = make_divisible(512 * scale)

        # PP-LCNet characteristic: GAP → 1x1 Conv(→1280) → H-Swish → Dropout
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_conv = nn.Conv2d(
            in_channels=last_out,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.last_act = nn.Hardswish(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        self.out_channels = 1280

    def _make_stage(self, stage_config):
        """Build a sequential stage from config list."""
        layers = []
        for k, in_c, out_c, s, se in stage_config:
            layers.append(
                PPLCNetBlock(
                    in_channels=make_divisible(in_c * self.scale),
                    out_channels=make_divisible(out_c * self.scale),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_act(x)

        # Feature extraction stages
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        # Head projection (PP-LCNet characteristic)
        x = self.avg_pool(x)  # → [B, C, 1, 1]
        x = self.last_conv(x)  # → [B, 1280, 1, 1]
        x = self.last_act(x)
        x = self.dropout(x)

        return x
