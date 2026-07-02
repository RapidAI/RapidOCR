# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
#
# PyTorch implementation of PPLCNetV4 backbone for PP-OCRv6.
# Converted from PaddlePaddle: PaddleOCR/ppocr/modeling/backbones/rec_lcnetv4.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Network configuration tables (identical to PaddleOCR)
# ---------------------------------------------------------------------------

NET_CONFIG_DET = {
    "tiny": {
        # stem(mid=16, out=32)  channels: 32 -> 48 -> 64 -> 160
        "stem": (16, 32),
        "blocks_s1": [[3, 32, 32, 1, True], [3, 32, 32, 1, False]],
        "blocks_s2": [
            [3, 32, 48, 2, False],
            [3, 48, 48, 1, True],
            [3, 48, 48, 1, False],
        ],
        "blocks_s3": [
            [3, 48, 64, 2, False],
            [3, 64, 64, 1, True],
            [3, 64, 64, 1, False],
            [3, 64, 64, 1, True],
            [3, 64, 64, 1, False],
        ],
        "blocks_s4": [
            [3, 64, 160, 2, False],
            [3, 160, 160, 1, True],
            [3, 160, 160, 1, False],
        ],
    },
    "small": {
        # stem(mid=24, out=48)  channels: 48 -> 96 -> 192 -> 384
        "stem": (24, 48),
        "blocks_s1": [[3, 48, 48, 1, True], [3, 48, 48, 1, False]],
        "blocks_s2": [
            [3, 48, 96, 2, False],
            [3, 96, 96, 1, True],
            [3, 96, 96, 1, False],
        ],
        "blocks_s3": [
            [3, 96, 192, 2, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
        ],
        "blocks_s4": [
            [3, 192, 384, 2, False],
            [3, 384, 384, 1, True],
            [3, 384, 384, 1, False],
        ],
    },
    "medium": {
        # stem(mid=64, out=128)  channels: 128 -> 256 -> 512 -> 896
        "stem": (64, 128),
        "blocks_s1": [[3, 128, 128, 1, True], [3, 128, 128, 1, False]],
        "blocks_s2": [
            [3, 128, 256, 2, False],
            [3, 256, 256, 1, True],
            [3, 256, 256, 1, False],
        ],
        "blocks_s3": [
            [3, 256, 512, 2, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
        ],
        "blocks_s4": [
            [3, 512, 896, 2, False],
            [3, 896, 896, 1, True],
            [3, 896, 896, 1, False],
        ],
    },
}


NET_CONFIG_REC = {
    "tiny": {
        # stem: simple (2xConv2D_BN+GELU, mid=24, out=48)  channels: 48 -> 96 -> 160
        "stem": (24, 48),
        "stem_type": "simple",
        "blocks2": [[3, 48, 48, 1, True]],
        "blocks3": [[3, 48, 48, 1, False]],
        "blocks4": [
            [3, 48, 96, (2, 1), False],
            [3, 96, 96, 1, True],
            [3, 96, 96, 1, False],
        ],
        "blocks5": [
            [3, 96, 160, (2, 1), False],
            [3, 160, 160, 1, True],
            [3, 160, 160, 1, False],
            [3, 160, 160, 1, False],
        ],
        "blocks6": [],
    },
    "small": {
        # stem: branch StemBlock (mid=48, out=96)  channels: 96 -> 192 -> 384
        "stem": (48, 96),
        "stem_type": "branch",
        "blocks2": [[3, 96, 96, 1, True]],
        "blocks3": [[3, 96, 96, 1, False], [3, 96, 96, 1, False]],
        "blocks4": [
            [3, 96, 192, (2, 1), False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
        ],
        "blocks5": [
            [3, 192, 384, (2, 1), False],
            [3, 384, 384, 1, True],
            [3, 384, 384, 1, False],
        ],
        "blocks6": [],
    },
    "medium": {
        # stem: branch StemBlock (mid=64, out=128)  channels: 128 -> 256 -> 512 -> 768
        "stem": (64, 128),
        "stem_type": "branch",
        "blocks2": [[3, 128, 128, 1, True]],
        "blocks3": [
            [3, 128, 256, 1, False],
            [3, 256, 256, 1, False],
            [3, 256, 256, 1, True],
        ],
        "blocks4": [
            [3, 256, 512, (2, 1), False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
        ],
        "blocks5": [
            [3, 512, 768, (2, 1), False],
            [3, 768, 768, 1, True],
            [3, 768, 768, 1, False],
        ],
        "blocks6": [],
    },
}

# ---------------------------------------------------------------------------
# PyTorch building blocks
# ---------------------------------------------------------------------------


class Conv2D_BN(nn.Sequential):
    """Conv2D + BatchNorm2d (bias=False).  Supports fuse() for deployment."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        bn_weight_init=1.0,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
        )
        bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(bn.weight, 1.0 if bn_weight_init == 1.0 else 0.0)
        nn.init.constant_(bn.bias, 0.0)
        self.add_module("bn", bn)

    @torch.no_grad()
    def fuse(self):
        """Fuse Conv2d + BatchNorm2d into a single Conv2d (bias=True)."""
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        # Handle 'same' padding
        if isinstance(c.padding, str):
            padding = (c.kernel_size[0] - 1) // 2
        else:
            padding = c.padding
        m = nn.Conv2d(
            w.shape[1] * c.groups,
            w.shape[0],
            w.shape[2:],
            stride=c.stride,
            padding=padding,
            groups=c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class ConvBNAct(nn.Module):
    """Conv2d + BN + ReLU.  Supports rep() that fuses Conv+BN into one Conv."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        use_act=True,
        lr_mult=1.0,
    ):
        super().__init__()
        self.use_act = use_act
        self.is_repped = False
        if isinstance(padding, str) and padding.upper() == "SAME":
            # Use PyTorch's native 'same' padding (handles asymmetric case correctly)
            padding = "same"
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if not self.is_repped:
            x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x

    @torch.no_grad()
    def rep(self):
        """Fuse Conv2d + BatchNorm2d into a single Conv2d (bias=True)."""
        if self.is_repped:
            return
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        fused_w = c.weight * w[:, None, None, None]
        fused_b = (
            bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        )
        # Handle 'same' padding
        if isinstance(c.padding, str):
            padding = (
                c.padding[0]
                if isinstance(c.padding, tuple)
                else (c.kernel_size[0] - 1) // 2
            )
        else:
            padding = c.padding
        m = nn.Conv2d(
            c.in_channels,
            c.out_channels,
            c.kernel_size,
            stride=c.stride,
            padding=padding,
            dilation=c.dilation,
            groups=c.groups,
        )
        m.weight.data.copy_(fused_w)
        m.bias.data.copy_(fused_b)
        self.conv = m
        del self.bn
        self.is_repped = True


class StemBlock(nn.Module):
    """Multi-branch stem with total stride 4 (stem1 stride=2 + stem3 stride=2)."""

    def __init__(self, in_channels=3, mid_channels=48, out_channels=96, lr_mult=1.0):
        super().__init__()
        self.is_repped = False
        self.stem1 = ConvBNAct(
            in_channels, mid_channels, 3, 2, use_act=True, lr_mult=lr_mult
        )
        self.stem2a = ConvBNAct(
            mid_channels,
            mid_channels // 2,
            2,
            1,
            padding="SAME",
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem2b = ConvBNAct(
            mid_channels // 2,
            mid_channels,
            2,
            1,
            padding="SAME",
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem3 = ConvBNAct(
            mid_channels * 2, mid_channels, 3, 2, use_act=True, lr_mult=lr_mult
        )
        self.stem4 = ConvBNAct(
            mid_channels, out_channels, 1, 1, padding=0, use_act=True, lr_mult=lr_mult
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        # Note: Paddle uses padding="SAME" (asymmetric), we use explicit F.pad

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2b(self.stem2a(x))
        # Explicit padding to match Paddle's MaxPool2D with padding="SAME"
        # kernel=2, stride=1, "SAME" -> pad (right=1, bottom=1) only
        x_pool = F.pad(x, [0, 1, 0, 1])
        x1 = self.pool(x_pool)
        x = self.stem4(self.stem3(torch.cat([x1, x2], dim=1)))
        return x

    def rep(self, fuse_lab=None):
        if self.is_repped:
            return
        for attr in ("stem1", "stem2a", "stem2b", "stem3", "stem4"):
            getattr(self, attr).rep()
        self.is_repped = True


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer (reduction=4, Hardsigmoid gating)."""

    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = x.mean(dim=[2, 3], keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.hardsigmoid(self.conv2(x))
        return x * identity


class RepDWConv(nn.Module):
    """Reparameterizable depthwise convolution.

    Training: 3-branch (3x3 DW + 1x1 DW + identity BN)
    Inference: fused into a single 3x3 DW Conv
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        self.conv = Conv2D_BN(
            channels, channels, kernel_size, 1, padding, groups=channels
        )
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

        self.is_repped = False
        self.reparam_conv = None

    def forward(self, x):
        if self.is_repped:
            return self.reparam_conv(x)
        return self.bn(self.conv(x) + self.conv1(x) + x)

    def rep(self, fuse_lab=None):
        if self.is_repped:
            return
        fused = self._fuse_conv()
        padding = (self.kernel_size - 1) // 2
        self.reparam_conv = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            1,
            padding,
            groups=self.channels,
        )
        self.reparam_conv.weight.data.copy_(fused.weight)
        self.reparam_conv.bias.data.copy_(fused.bias)
        del self.conv
        del self.conv1
        del self.bn
        self.is_repped = True

    @torch.no_grad()
    def _fuse_conv(self):
        conv = self.conv.fuse()
        pad_size = self.kernel_size // 2
        conv1_w = F.pad(self.conv1.weight, [pad_size, pad_size, pad_size, pad_size])
        identity = F.pad(
            torch.ones(
                [self.conv1.weight.shape[0], self.conv1.weight.shape[1], 1, 1],
                device=self.conv1.weight.device,
                dtype=self.conv1.weight.dtype,
            ),
            [pad_size, pad_size, pad_size, pad_size],
        )
        w = conv.weight + conv1_w + identity
        conv.weight.data.copy_(w)
        bn = self.bn
        scale = bn.weight / (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(conv.weight * scale[:, None, None, None])
        conv.bias.data.copy_(bn.bias + (conv.bias - bn.running_mean) * scale)
        return conv


class LCNetV4Block(nn.Module):
    """LCNetV4 block for detection and recognition.

    Token mixer: RepDWConv when stride=1 and in==out, else plain Conv2D_BN DW conv.
    Channel mixer: expand -> act -> compress (+ residual when stride=1 and in==out)
    rep() fuses all Conv2D_BN layers (mathematically exact, no accuracy change).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        lr_mult=1.0,
        expand_ratio=2,
        act_type="gelu",
    ):
        super().__init__()
        self.is_repped = False
        self.has_residual = in_channels == out_channels and stride == 1
        self.use_rep_dw = stride == 1 and in_channels == out_channels

        self.token_mixer = nn.Sequential()
        if self.use_rep_dw:
            self.token_mixer.add_module("rep_dw", RepDWConv(in_channels, dw_size))
        else:
            padding = (dw_size - 1) // 2
            self.token_mixer.add_module(
                "dw_conv",
                Conv2D_BN(
                    in_channels,
                    in_channels,
                    dw_size,
                    stride,
                    padding,
                    groups=in_channels,
                ),
            )
        if use_se:
            self.token_mixer.add_module("se", SELayer(in_channels, lr_mult=lr_mult))

        hidden_channels = int(in_channels * expand_ratio)
        compress_bn_init = 0.0 if self.has_residual else 1.0
        self.channel_mixer = nn.Sequential()
        self.channel_mixer.add_module(
            "expand", Conv2D_BN(in_channels, hidden_channels, 1, 1, 0)
        )
        if act_type == "gelu":
            self.channel_mixer.add_module("act", nn.GELU())
        elif act_type == "hswish":
            self.channel_mixer.add_module("act", nn.Hardswish())
        elif act_type == "relu":
            self.channel_mixer.add_module("act", nn.ReLU(inplace=True))
        self.channel_mixer.add_module(
            "compress",
            Conv2D_BN(
                hidden_channels,
                out_channels,
                1,
                1,
                0,
                bn_weight_init=compress_bn_init,
            ),
        )

    def forward(self, x):
        x = self.token_mixer(x)
        if self.has_residual:
            return x + self.channel_mixer(x)
        return self.channel_mixer(x)

    def rep(self, fuse_lab=None):
        if self.is_repped:
            return
        if self.use_rep_dw:
            self.token_mixer.rep_dw.rep(fuse_lab=fuse_lab)
        else:
            self.token_mixer.dw_conv = self.token_mixer.dw_conv.fuse()
        for name in ("expand", "compress"):
            m = getattr(self.channel_mixer, name, None)
            if isinstance(m, Conv2D_BN):
                setattr(self.channel_mixer, name, m.fuse())
        self.is_repped = True


# ---------------------------------------------------------------------------
# PPLCNetV4 – unified backbone for det & rec
# ---------------------------------------------------------------------------


class PPLCNetV4(nn.Module):
    """Unified PPLCNetV4 backbone for text detection and recognition.

    Detection (det=True):
        model_size in {'tiny', 'small', 'medium'} -- see NET_CONFIG_DET.
        Returns 4-level feature list [s1_out, s2_out, s3_out, s4_out].

    Recognition (det=False):
        model_size in {'tiny', 'small', 'medium'} -- see NET_CONFIG_REC.
        Returns pooled feature tensor [B, C, 1, W].
    """

    def __init__(
        self,
        det=False,
        model_size="small",
        in_channels=3,
        lr_mult_list=None,
        **kwargs,
    ):
        super().__init__()
        if lr_mult_list is None:
            lr_mult_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.det = det
        self.is_repped = False

        if det:
            assert (
                model_size in NET_CONFIG_DET
            ), "det model_size must be one of {} but got '{}'".format(
                list(NET_CONFIG_DET.keys()), model_size
            )
            cfg = NET_CONFIG_DET[model_size]
            stem_mid, stem_out = cfg["stem"]
            self.stem = StemBlock(in_channels, stem_mid, stem_out)

            def make_stage(key):
                return nn.Sequential(
                    *[
                        LCNetV4Block(in_c, out_c, s, k, se, expand_ratio=2)
                        for k, in_c, out_c, s, se in cfg[key]
                    ]
                )

            self.blocks_s1 = make_stage("blocks_s1")
            self.blocks_s2 = make_stage("blocks_s2")
            self.blocks_s3 = make_stage("blocks_s3")
            self.blocks_s4 = make_stage("blocks_s4")
            self.out_channels = [
                cfg["blocks_s1"][-1][2],
                cfg["blocks_s2"][-1][2],
                cfg["blocks_s3"][-1][2],
                cfg["blocks_s4"][-1][2],
            ]

        else:
            assert isinstance(lr_mult_list, (list, tuple)) and len(lr_mult_list) == 6
            assert (
                model_size in NET_CONFIG_REC
            ), "rec model_size must be one of {} but got '{}'".format(
                list(NET_CONFIG_REC.keys()), model_size
            )
            self.lr_mult_list = lr_mult_list
            cfg = NET_CONFIG_REC[model_size]
            stem_mid, stem_out = cfg["stem"]
            if cfg["stem_type"] == "branch":
                self.conv1 = StemBlock(3, stem_mid, stem_out, lr_mult=lr_mult_list[0])
            else:
                self.conv1 = nn.Sequential(
                    Conv2D_BN(3, stem_mid, 3, 2, 1),
                    nn.GELU(),
                    Conv2D_BN(stem_mid, stem_out, 3, 2, 1),
                )

            def make_stage(stage_name, lr_idx):
                return nn.Sequential(
                    *[
                        LCNetV4Block(
                            in_c,
                            out_c,
                            s,
                            k,
                            se,
                            lr_mult=lr_mult_list[lr_idx],
                            expand_ratio=2,
                        )
                        for k, in_c, out_c, s, se in cfg.get(stage_name, [])
                    ]
                )

            self.blocks2 = make_stage("blocks2", 1)
            self.blocks3 = make_stage("blocks3", 2)
            self.blocks4 = make_stage("blocks4", 3)
            self.blocks5 = make_stage("blocks5", 4)
            self.blocks6 = make_stage("blocks6", 5)

            for sname in reversed(
                ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
            ):
                if cfg.get(sname):
                    self.out_channels = cfg[sname][-1][2]
                    break

    def forward(self, x):
        if self.det:
            x = self.stem(x)
            o1 = self.blocks_s1(x)
            o2 = self.blocks_s2(o1)
            o3 = self.blocks_s3(o2)
            o4 = self.blocks_s4(o3)
            return [o1, o2, o3, o4]
        else:
            x = self.conv1(x)
            x = self.blocks2(x)
            x = self.blocks3(x)
            x = self.blocks4(x)
            x = self.blocks5(x)
            x = self.blocks6(x)
            if self.training:
                x = F.adaptive_avg_pool2d(x, [1, 40])
            else:
                assert x.shape[2] >= 3, f"Feature height {x.shape[2]} < pool kernel 3."
                x = F.avg_pool2d(x, [3, 2])
            return x

    def rep(self, fuse_lab=None):
        if self.is_repped:
            return
        if self.det:
            self.stem.rep()
            for stage in [
                self.blocks_s1,
                self.blocks_s2,
                self.blocks_s3,
                self.blocks_s4,
            ]:
                for block in stage:
                    block.rep(fuse_lab=fuse_lab)
        else:
            if hasattr(self.conv1, "rep"):
                self.conv1.rep()
            for stage in [
                self.blocks2,
                self.blocks3,
                self.blocks4,
                self.blocks5,
                self.blocks6,
            ]:
                for block in stage:
                    block.rep()
        self.is_repped = True
