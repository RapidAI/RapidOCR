import torch
import torch.nn.functional as F
from torch import nn

from ..backbones.det_mobilenet_v3 import SEModule
from .intracl import IntraCLBlock


def hard_swish(x, inplace=True):
    return x * F.relu6(x + 3.0, inplace=inplace) / 6.0


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=1,
        groups=None,
        if_act=True,
        act="relu",
        **kwargs,
    ):
        super(DSConv, self).__init__()
        if groups is None:
            groups = in_channels

        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(int(in_channels * 4))

        self.conv3 = nn.Conv2d(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = hard_swish(x)
            else:
                print(
                    "The activation function({}) is selected incorrectly.".format(
                        self.act
                    )
                )
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class DBFPN(nn.Module):
    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf

        self.in2_conv = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.in3_conv = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.in4_conv = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.in5_conv = nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.p5_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.p4_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.p3_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.p2_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.interpolate(
            in5,
            scale_factor=2,
            mode="nearest",
        )  # align_mode=1)  # 1/16
        out3 = in3 + F.interpolate(
            out4,
            scale_factor=2,
            mode="nearest",
        )  # align_mode=1)  # 1/8
        out2 = in2 + F.interpolate(
            out3,
            scale_factor=2,
            mode="nearest",
        )  # align_mode=1)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.interpolate(
            p5,
            scale_factor=8,
            mode="nearest",
        )  # align_mode=1)
        p4 = F.interpolate(
            p4,
            scale_factor=4,
            mode="nearest",
        )  # align_mode=1)
        p3 = F.interpolate(
            p3,
            scale_factor=2,
            mode="nearest",
        )  # align_mode=1)

        fuse = torch.cat([p5, p4, p3, p2], dim=1)

        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        return fuse


class RSELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False,
        )
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.intracl = False
        if "intracl" in kwargs.keys() and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut)
            )
            self.inp_conv.append(
                RSELayer(
                    out_channels, out_channels // 4, kernel_size=3, shortcut=shortcut
                )
            )

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class LKPAN(nn.Module):
    def __init__(self, in_channels, out_channels, mode="large", **kwargs):
        super(LKPAN, self).__init__()
        self.out_channels = out_channels

        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        # pan head
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()

        if mode.lower() == "lite":
            p_layer = DSConv
        elif mode.lower() == "large":
            p_layer = nn.Conv2d
        else:
            raise ValueError(
                "mode can only be one of ['lite', 'large'], but received {}".format(
                    mode
                )
            )

        for i in range(len(in_channels)):
            self.ins_conv.append(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    bias=False,
                )
            )

            self.inp_conv.append(
                p_layer(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    bias=False,
                )
            )

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2d(
                        in_channels=self.out_channels // 4,
                        out_channels=self.out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                )
            self.pan_lat_conv.append(
                p_layer(
                    in_channels=self.out_channels // 4,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    bias=False,
                )
            )
            self.intracl = False
            if "intracl" in kwargs.keys() and kwargs["intracl"] is True:
                self.intracl = kwargs["intracl"]
                self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
                self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
                self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
                self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class ASFBlock(nn.Module):
    """
    This code is refered from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)

        self.spatial_scale = nn.Sequential(
            # Nx1xHxW
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                bias=False,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        self.channel_scale = nn.Sequential(
            nn.Conv2d(
                in_channels=inter_channels,
                out_channels=out_features_num,
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = torch.mean(fuse_features, dim=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i : i + 1] * features_list[i])
        return torch.cat(out_list, dim=1)


# ============================================================================
#  PP-OCRv6 new components: DilatedReparamBlock, RepLKFPN, RepLKPAN
# ============================================================================


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block from UniRepLKNet.
    Reference: https://github.com/AILab-CVC/UniRepLKNet

    Training: uses multiple parallel dilated depthwise convolutions + BN
    Inference: all branches merge into a single large-kernel depthwise conv

    For kernel_size=7, the branches are:
      - origin: 7x7 DW Conv (dil=1)
      - branch1: 5x5 DW Conv (dil=1, equiv RF=5)
      - branch2: 3x3 DW Conv (dil=2, equiv RF=5)
      - branch3: 3x3 DW Conv (dil=3, equiv RF=7)
    """

    def __init__(self, channels, kernel_size=7, deploy=False):
        super(DilatedReparamBlock, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.is_repped = deploy

        if kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        else:
            raise ValueError(
                "DilatedReparamBlock requires kernel_size in [5,7,9,11,13], "
                "but got {}".format(kernel_size)
            )

        if not self.is_repped:
            self.lk_origin = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            )
            self.origin_bn = nn.BatchNorm2d(channels)

            for k, r in zip(self.kernel_sizes, self.dilates):
                equiv_ks = r * (k - 1) + 1
                p = equiv_ks // 2
                conv = nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=k,
                    stride=1,
                    padding=p,
                    dilation=r,
                    groups=channels,
                    bias=False,
                )
                bn = nn.BatchNorm2d(channels)
                setattr(self, "dil_conv_k{}_{}".format(k, r), conv)
                setattr(self, "dil_bn_k{}_{}".format(k, r), bn)
        else:
            self.lk_origin = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias=True,
            )

    def forward(self, x):
        if self.is_repped:
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, "dil_conv_k{}_{}".format(k, r))
            bn = getattr(self, "dil_bn_k{}_{}".format(k, r))
            out = out + bn(conv(x))
        return out

    @staticmethod
    def _fuse_bn(conv, bn):
        """Fuse Conv2d + BatchNorm2d into (weight, bias)."""
        kernel = conv.weight
        gamma = bn.weight
        beta = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        fused_weight = kernel * (gamma / std).reshape([-1, 1, 1, 1])
        fused_bias = beta - running_mean * gamma / std
        return fused_weight, fused_bias

    @staticmethod
    def _convert_dilated_to_nondilated(kernel, dilate_rate):
        """Convert dilated conv kernel to equivalent non-dilated (sparse) kernel
        by inserting zeros using transposed convolution."""
        if dilate_rate == 1:
            return kernel
        identity = torch.ones([1, 1, 1, 1], dtype=kernel.dtype, device=kernel.device)
        # F.conv_transpose2d with stride=dilate_rate inserts zeros
        C = kernel.shape[0]
        result_list = []
        for i in range(C):
            k_i = kernel[i : i + 1]  # (1, 1, kH, kW)
            dilated = F.conv_transpose2d(k_i, identity, stride=dilate_rate)
            result_list.append(dilated)
        return torch.cat(result_list, dim=0)

    @staticmethod
    def _merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
        """Pad dilated equivalent kernel to large kernel size and add."""
        large_k = large_kernel.shape[2]
        dilated_k = dilated_kernel.shape[2]
        equiv_ks = dilated_r * (dilated_k - 1) + 1
        equiv_kernel = DilatedReparamBlock._convert_dilated_to_nondilated(
            dilated_kernel, dilated_r
        )
        rows_to_pad = large_k // 2 - equiv_ks // 2
        if rows_to_pad > 0:
            merged = large_kernel + F.pad(
                equiv_kernel, [rows_to_pad, rows_to_pad, rows_to_pad, rows_to_pad]
            )
        else:
            merged = large_kernel + equiv_kernel
        return merged

    @torch.no_grad()
    def rep(self):
        """Merge all parallel branches into a single large-kernel DW conv."""
        if self.is_repped:
            return
        origin_k, origin_b = self._fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, "dil_conv_k{}_{}".format(k, r))
            bn = getattr(self, "dil_bn_k{}_{}".format(k, r))
            branch_k, branch_b = self._fuse_bn(conv, bn)
            origin_k = self._merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b = origin_b + branch_b

        merged_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias=True,
        )
        merged_conv.weight.data.copy_(origin_k)
        merged_conv.bias.data.copy_(origin_b)
        self.lk_origin = merged_conv
        self.is_repped = True

        delattr(self, "origin_bn")
        for k, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, "dil_conv_k{}_{}".format(k, r))
            delattr(self, "dil_bn_k{}_{}".format(k, r))


class DilatedReparamConv(nn.Module):
    """
    A drop-in replacement for standard Conv2d (in_ch -> out_ch, large kernel)
    using DilatedReparamBlock (depthwise) + 1x1 pointwise convolution.

    Architecture:
      input(in_ch) -> DilatedReparamBlock(in_ch, DW, kernel_size) -> 1x1 Conv(in_ch->out_ch) -> BN
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=9, deploy=False, **kwargs
    ):
        super(DilatedReparamConv, self).__init__()
        self.is_repped = False
        self.dw = DilatedReparamBlock(
            channels=in_channels, kernel_size=kernel_size, deploy=deploy
        )
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        if not self.is_repped:
            x = self.bn(x)
        return x

    @torch.no_grad()
    def rep(self):
        """Fuse DW branches + PW Conv + BN for deployment."""
        if self.is_repped:
            return
        self.dw.rep()
        # Fuse pw(Conv2d, no bias) + bn(BatchNorm2d) into single Conv2d with bias
        conv, bn = self.pw, self.bn
        gamma = bn.weight
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = gamma / std
        w = conv.weight * scale[:, None, None, None]
        b = bn.bias - bn.running_mean * scale
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )
        fused.weight.data.copy_(w)
        fused.bias.data.copy_(b)
        self.pw = fused
        del self.bn
        self.is_repped = True


class RepLKFPN(nn.Module):
    """Optimized RSEFPN: replaces 3x3 standard Conv in inp_conv with
    DilatedReparamBlock (DW, kernel_size) + PWConv 1x1 + SE.

    Changes vs RSEFPN:
      - inp_conv: RSELayer(3x3 std Conv + SE)
                -> DilatedReparamBlock(DW kernel_size) + PWConv(1x1) + SE
      - ins_conv: unchanged (1x1 Conv, no benefit from DW decomposition)
    """

    def __init__(
        self, in_channels, out_channels, shortcut=True, dilated_kernel_size=7, **kwargs
    ):
        super(RepLKFPN, self).__init__()
        self.out_channels = out_channels
        self.is_repped = False
        self.ins_conv = nn.ModuleList()
        self.inp_conv_dw = nn.ModuleList()
        self.inp_conv_pw = nn.ModuleList()
        self.inp_conv_se = nn.ModuleList()
        self.shortcut = shortcut

        self.intracl = False
        if "intracl" in kwargs and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut)
            )

            self.inp_conv_dw.append(
                DilatedReparamBlock(
                    channels=out_channels, kernel_size=dilated_kernel_size
                )
            )

            self.inp_conv_pw.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels // 4,
                    kernel_size=1,
                    bias=False,
                )
            )

            self.inp_conv_se.append(SEModule(out_channels // 4))

    def _inp_forward(self, x, idx):
        x = self.inp_conv_dw[idx](x)
        x = self.inp_conv_pw[idx](x)
        if self.shortcut:
            x = x + self.inp_conv_se[idx](x)
        else:
            x = self.inp_conv_se[idx](x)
        return x

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        p5 = self._inp_forward(in5, 3)
        p4 = self._inp_forward(out4, 2)
        p3 = self._inp_forward(out3, 1)
        p2 = self._inp_forward(out2, 0)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse

    def rep(self):
        """Merge DilatedReparamBlock branches for inference deployment."""
        if self.is_repped:
            return
        for i in range(len(self.inp_conv_dw)):
            self.inp_conv_dw[i].rep()
        self.is_repped = True


class RepLKPAN(nn.Module):
    """
    Optimized LKPAN using UniRepLKNet's DilatedReparamBlock.

    Replaces the 8 standard 9x9 Conv2d in LKPAN (4 inp_conv + 4 pan_lat_conv)
    with DilatedReparamConv (DW large-kernel reparam + 1x1 pointwise).
    """

    def __init__(self, in_channels, out_channels, mode="large", **kwargs):
        super(RepLKPAN, self).__init__()
        self.out_channels = out_channels
        self.is_repped = False

        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        # pan head
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    bias=False,
                )
            )

            self.inp_conv.append(
                DilatedReparamConv(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                )
            )

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2d(
                        in_channels=self.out_channels // 4,
                        out_channels=self.out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                )
            self.pan_lat_conv.append(
                DilatedReparamConv(
                    in_channels=self.out_channels // 4,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                )
            )

        self.intracl = False
        if "intracl" in kwargs and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse

    def rep(self):
        """Merge all DilatedReparamBlock branches and fuse PW+BN for deployment."""
        if self.is_repped:
            return
        for i in range(len(self.inp_conv)):
            self.inp_conv[i].rep()
        for i in range(len(self.pan_lat_conv)):
            self.pan_lat_conv[i].rep()
        self.is_repped = True
