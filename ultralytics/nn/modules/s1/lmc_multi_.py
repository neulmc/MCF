from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from .conv import Conv, autopad
import torch.nn.functional as F

__all__ = (
    "Identity",
    "InputSel",
    "TripleDEA",
    "PDCC2f",
)

class Identity(nn.Module):
    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class InputSel(nn.Module):
    def __init__(self, c1, c2, mode, *args, **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'rgb':
            return x[:, 0:3, :, :]
        elif self.mode == 'depth1':
            return x[:, 3:6, :, :]
        elif self.mode == 'ir':
            return x[:, 6:, :, :]

class lmcTripleDEA(nn.Module):
    """三模态 DEA：RGB + Depth + IR"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = lmcTripleDECA(channel, kernel_size, p_kernel, reduction)
        self.depa = lmcTripleDEPA(channel, m_kernel)
        #self.act = nn.Sigmoid()

    def forward(self, x_rgb_depth_ir):
        # DECA：通道增强
        x_rgb, x_depth, x_ir = x_rgb_depth_ir[0], x_rgb_depth_ir[1], x_rgb_depth_ir[2]
        r_rgb, r_depth, r_ir = self.deca(x_rgb, x_depth, x_ir)
        # DEPA：空间增强
        r_rgb, r_depth, r_ir = self.depa(r_rgb, r_depth, r_ir)
        # 三模态融合输出
        #fused = self.act(r_rgb + r_depth + r_ir)
        return r_rgb + r_depth + r_ir

class lmcTripleDECA(nn.Module):
    """RGB + Depth + IR 三模态通道增强"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size

        # 三个模态各自提取通道权重（SENet风格）
        self.fc_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            #nn.Sigmoid()
        )
        self.fc_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            #nn.Sigmoid()
        )
        self.fc_ir = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            #nn.Sigmoid()
        )

        # 跨模态混合：3个模态拼接 → 压缩回 channel
        self.compress = Conv(channel * 3, channel, 3)

        # 多尺度卷积金字塔
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel),
            nn.SiLU()
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel),
            nn.SiLU()
        )
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(
                channel, channel,
                int(self.kernel_size / kernel1 / kernel2),
                int(self.kernel_size / kernel1 / kernel2),
                0, groups=channel
            ),
            nn.SiLU()
        )
        self.act = nn.Sigmoid()

    def forward(self, x_rgb, x_depth, x_ir):
        b, c, h, w = x_rgb.size()

        # 1. 各模态通道权重
        w_rgb = self.fc_rgb(x_rgb).view(b, c, 1, 1)
        w_depth = self.fc_depth(x_depth).view(b, c, 1, 1)
        w_ir = self.fc_ir(x_ir).view(b, c, 1, 1)

        # 2. 三模态融合的全局特征
        glob_t = self.compress(torch.cat([x_rgb, x_depth, x_ir], 1))
        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        # 3. 三方交叉增强
        # RGB ← 从 Depth 和 IR 借通道重要性
        w_for_rgb = self.act(w_depth * glob + w_ir * glob)
        # Depth ← 从 RGB 和 IR 借
        w_for_depth = self.act(w_rgb * glob + w_ir * glob)
        # IR ← 从 RGB 和 Depth 借
        w_for_ir = self.act(w_rgb * glob + w_depth * glob)

        result_rgb = x_rgb * w_for_rgb
        result_depth = x_depth * w_for_depth
        result_ir = x_ir * w_for_ir

        return result_rgb, result_depth, result_ir

class lmcTripleDEPA(nn.Module):
    """RGB + Depth + IR 三模态空间增强"""

    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        if m_kernel is None:
            m_kernel = [3, 7]

        # 每个模态用两个不同卷积核提取空间权重
        self.cv_rgb1 = Conv(channel, 1, m_kernel[0])
        self.cv_rgb2 = Conv(channel, 1, m_kernel[1])
        self.cv_depth1 = Conv(channel, 1, m_kernel[0])
        self.cv_depth2 = Conv(channel, 1, m_kernel[1])
        self.cv_ir1 = Conv(channel, 1, m_kernel[0])
        self.cv_ir2 = Conv(channel, 1, m_kernel[1])

        # 合并卷积：2通道 → 1通道
        self.merge = Conv(2, 1, 5)

        # 各模态压缩到1通道
        self.compress_rgb = Conv(channel, 1, 3)
        self.compress_depth = Conv(channel, 1, 3)
        self.compress_ir = Conv(channel, 1, 3)

        self.act = nn.Sigmoid()

    def forward(self, x_rgb, x_depth, x_ir):
        # 1. 各模态空间权重（多尺度融合）
        w_rgb = self.merge(torch.cat([self.cv_rgb1(x_rgb), self.cv_rgb2(x_rgb)], 1))
        w_depth = self.merge(torch.cat([self.cv_depth1(x_depth), self.cv_depth2(x_depth)], 1))
        w_ir = self.merge(torch.cat([self.cv_ir1(x_ir), self.cv_ir2(x_ir)], 1))

        # 2. 三模态空间融合
        #glob = self.act(self.compress_rgb(x_rgb) + self.compress_depth(x_depth) + self.compress_ir(x_ir))
        glob = self.compress_rgb(x_rgb) + self.compress_depth(x_depth) + self.compress_ir(x_ir)

        # 3. 三方交叉增强：每个模态用另外两个模态的空间权重
        w_for_rgb = self.act(glob + w_depth + w_ir)  # RGB 用 Depth+IR 增强
        w_for_depth = self.act(glob + w_rgb + w_ir)  # Depth 用 RGB+IR 增强
        w_for_ir = self.act(glob + w_rgb + w_depth)  # IR 用 RGB+Depth 增强

        result_rgb = x_rgb * w_for_rgb
        result_depth = x_depth * w_for_depth
        result_ir = x_ir * w_for_ir

        return result_rgb, result_depth, result_ir

class TripleDEA(nn.Module):
    """三模态 DEA：RGB + Depth + IR"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = TripleDECA(channel, kernel_size, p_kernel, reduction)
        self.depa = TripleDEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x_rgb_depth_ir):
        # DECA：通道增强
        x_rgb, x_depth, x_ir = x_rgb_depth_ir[0], x_rgb_depth_ir[1], x_rgb_depth_ir[2]
        r_rgb, r_depth, r_ir = self.deca(x_rgb, x_depth, x_ir)
        # DEPA：空间增强
        r_rgb, r_depth, r_ir = self.depa(r_rgb, r_depth, r_ir)
        # 三模态融合输出
        fused = self.act(r_rgb + r_depth + r_ir)
        return fused

class TripleDECA(nn.Module):
    """RGB + Depth + IR 三模态通道增强"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size

        # 三个模态各自提取通道权重（SENet风格）
        self.fc_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc_ir = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # 跨模态混合：3个模态拼接 → 压缩回 channel
        self.compress = Conv(channel * 3, channel, 3)

        # 多尺度卷积金字塔
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel),
            nn.SiLU()
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel),
            nn.SiLU()
        )
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(
                channel, channel,
                int(self.kernel_size / kernel1 / kernel2),
                int(self.kernel_size / kernel1 / kernel2),
                0, groups=channel
            ),
            nn.SiLU()
        )
        self.act = nn.Sigmoid()

    def forward(self, x_rgb, x_depth, x_ir):
        b, c, h, w = x_rgb.size()

        # 1. 各模态通道权重
        w_rgb = self.fc_rgb(x_rgb).view(b, c, 1, 1)
        w_depth = self.fc_depth(x_depth).view(b, c, 1, 1)
        w_ir = self.fc_ir(x_ir).view(b, c, 1, 1)

        # 2. 三模态融合的全局特征
        glob_t = self.compress(torch.cat([x_rgb, x_depth, x_ir], 1))
        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        # 3. 三方交叉增强
        # RGB ← 从 Depth 和 IR 借通道重要性
        w_for_rgb = self.act(w_depth * glob + w_ir * glob)
        # Depth ← 从 RGB 和 IR 借
        w_for_depth = self.act(w_rgb * glob + w_ir * glob)
        # IR ← 从 RGB 和 Depth 借
        w_for_ir = self.act(w_rgb * glob + w_depth * glob)

        result_rgb = x_rgb * w_for_rgb
        result_depth = x_depth * w_for_depth
        result_ir = x_ir * w_for_ir

        return result_rgb, result_depth, result_ir

class TripleDEPA(nn.Module):
    """RGB + Depth + IR 三模态空间增强"""

    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        if m_kernel is None:
            m_kernel = [3, 7]

        # 每个模态用两个不同卷积核提取空间权重
        self.cv_rgb1 = Conv(channel, 1, m_kernel[0])
        self.cv_rgb2 = Conv(channel, 1, m_kernel[1])
        self.cv_depth1 = Conv(channel, 1, m_kernel[0])
        self.cv_depth2 = Conv(channel, 1, m_kernel[1])
        self.cv_ir1 = Conv(channel, 1, m_kernel[0])
        self.cv_ir2 = Conv(channel, 1, m_kernel[1])

        # 合并卷积：2通道 → 1通道
        self.merge = Conv(2, 1, 5)

        # 各模态压缩到1通道
        self.compress_rgb = Conv(channel, 1, 3)
        self.compress_depth = Conv(channel, 1, 3)
        self.compress_ir = Conv(channel, 1, 3)

        self.act = nn.Sigmoid()

    def forward(self, x_rgb, x_depth, x_ir):
        # 1. 各模态空间权重（多尺度融合）
        w_rgb = self.merge(torch.cat([self.cv_rgb1(x_rgb), self.cv_rgb2(x_rgb)], 1))
        w_depth = self.merge(torch.cat([self.cv_depth1(x_depth), self.cv_depth2(x_depth)], 1))
        w_ir = self.merge(torch.cat([self.cv_ir1(x_ir), self.cv_ir2(x_ir)], 1))

        # 2. 三模态空间融合
        glob = self.act(self.compress_rgb(x_rgb) + self.compress_depth(x_depth) + self.compress_ir(x_ir))

        # 3. 三方交叉增强：每个模态用另外两个模态的空间权重
        w_for_rgb = self.act(glob + w_depth + w_ir)  # RGB 用 Depth+IR 增强
        w_for_depth = self.act(glob + w_rgb + w_ir)  # Depth 用 RGB+IR 增强
        w_for_ir = self.act(glob + w_rgb + w_depth)  # IR 用 RGB+Depth 增强

        result_rgb = x_rgb * w_for_rgb
        result_depth = x_depth * w_for_depth
        result_ir = x_ir * w_for_ir

        return result_rgb, result_depth, result_ir

class PDCC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, mode: str='xxxx', e: float = 0.5, shortcut: bool = True, g: int = 1):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        #self.m = nn.ModuleList(PDCBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.m = nn.ModuleList(PDCBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3))) for _ in range(n)) # identity!!!! lmc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class PDCBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = MixedPDConv(c1, c_, k[0], 1)
        self.cv2 = MixedPDConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class MixedPDConv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = PConv2d(c1, c2, k, stride =s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

class PConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 alpha_init=0.0):
        super(PConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        #self.weight1 = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        #self.weight2 = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        #self.weight3 = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # alpha 可学习，先经 sigmoid 保证在 0~1 之间
        #self.alpha = nn.Parameter(torch.tensor(alpha_init))
        #self.alpha = nn.Parameter(torch.zeros(out_channels))
        #self.alpha_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        #self.alpha_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.alpha_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # 三种 pdc 函数
        self.pdc_cd = CDConv2d()
        self.pdc_ad = ADConv2d()
        self.pdc_rd = RDConv2d()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.weight3, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 强制转 FP32
        x_fp32 = x.float()
        weight_fp32 = self.weight.float()
        bias_fp32 = self.bias.float() if self.bias is not None else None

        # 普通卷积
        out_cv = F.conv2d(x_fp32, weight_fp32, bias_fp32, self.stride, self.padding, self.dilation, self.groups)

        # 三种差分 (共享 weight)
        out_cd = self.pdc_cd(x_fp32, weight_fp32, bias_fp32, self.stride, self.padding, self.dilation, self.groups)
        out_ad = self.pdc_ad(x_fp32, weight_fp32, bias_fp32, self.stride, self.padding, self.dilation, self.groups)
        out_rd = self.pdc_rd(x_fp32, weight_fp32, bias_fp32, self.stride, self.padding, self.dilation, self.groups)

        # 三个差分的平均
        out_diff = (out_cd + out_ad + out_rd) / 3.0

        # alpha
        alpha = torch.sigmoid(self.alpha_conv(x_fp32))  # [B, C, H, W]
        out = alpha * out_cv + (1 - alpha) * out_diff

        # 转回原 dtype
        return out.to(x.dtype)

class CDConv2d(nn.Module):
    """Central Difference Convolution"""
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
        assert weight.size(2) == 3 and weight.size(3) == 3, 'kernel size for cd_conv should be 3x3'
        pad_val = padding if isinstance(padding, int) else padding[0]
        assert pad_val == dilation, 'padding for cd_conv set wrong'

        weights_c = weight.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
        y = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y - yc

class ADConv2d(nn.Module):
    """Angular Difference Convolution"""
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
        assert weight.size(2) == 3 and weight.size(3) == 3, 'kernel size for ad_conv should be 3x3'
        pad_val = padding if isinstance(padding, int) else padding[0]
        assert pad_val == dilation, 'padding for ad_conv set wrong'

        shape = weight.shape
        w = weight.view(shape[0], shape[1], -1)
        w_ad = (w - w[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        return F.conv2d(x, w_ad, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

class RDConv2d(nn.Module):
    """Radial Difference Convolution"""
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
        assert weight.size(2) == 3 and weight.size(3) == 3, 'kernel size for rd_conv should be 3x3'

        rd_padding = 2 * dilation
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        w = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = w[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -w[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return F.conv2d(x, buffer, bias, stride=stride, padding=rd_padding, dilation=dilation, groups=groups)

## cd, ad, rd convolutions
'''
def createConvFunc(op_type):
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding[0] == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding[0] == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None
'''
