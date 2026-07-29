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
    "InputSel_m3data",
    "TriMFBlock",
    "BiMFBlock",
    "PDCC2f",
    "PDCC2fback",
    "TriMFBlocksub1",
    "TriMFBlocksub2",
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

class InputSel_m3data(nn.Module):
    def __init__(self, c1, c2, mode, *args, **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'rgb':
            return x[:, 0:3, :, :]
        elif self.mode == 'ir':
            return x[:, 3:, :, :]

class TriMFBlock(nn.Module):
    """三模态：RGB + Depth + IR"""

    def __init__(self, channel=(128,64,64), grid_size = 8, head_dim=16, m_kernel=None):
        super().__init__()
        # CrossModalFeatureTransform
        self.tmft = TriMFTBlock(channel, grid_size = grid_size, head_dim=head_dim)
        # CrossModalFeatureFusion
        self.tmff = TriMFFBlock(channel, m_kernel)
        self.c_rgb = channel[0]
        self.c_depth = channel[1]
        self.c_ir = channel[2]

    def forward(self, x_rgb_depth_ir):
        x_rgb, x_depth, x_ir = x_rgb_depth_ir[0], x_rgb_depth_ir[1], x_rgb_depth_ir[2]
        r_rgb, r_depth, r_ir = self.tmft(x_rgb, x_depth, x_ir)
        fused = self.tmff(r_rgb, r_depth, r_ir)
        return fused

class TriMFBlocksub1(nn.Module):
    """三模态：RGB + Depth + IR"""

    def __init__(self, channel=(128,64,64), grid_size = 8, head_dim=16, m_kernel=None):
        super().__init__()
        # CrossModalFeatureTransform
        self.tmft = TriMFTBlock(channel, grid_size = grid_size, head_dim=head_dim)
        # CrossModalFeatureFusion
        #self.tmff = TriMFFBlock(channel, m_kernel)
        self.proj = Conv(channel[0] + channel[1] + channel[2], channel[0], 1)
        self.c_rgb = channel[0]
        self.c_depth = channel[1]
        self.c_ir = channel[2]

    def forward(self, x_rgb_depth_ir):
        x_rgb, x_depth, x_ir = x_rgb_depth_ir[0], x_rgb_depth_ir[1], x_rgb_depth_ir[2]
        r_rgb, r_depth, r_ir = self.tmft(x_rgb, x_depth, x_ir)
        fused = torch.cat([r_rgb, r_depth, r_ir], dim=1)  # [B, c_rgb+c_depth+c_ir, H, W]
        return self.proj(fused)

class TriMFBlocksub2(nn.Module):
    """三模态：RGB + Depth + IR"""

    def __init__(self, channel=(128,64,64), grid_size = 8, head_dim=16, m_kernel=None):
        super().__init__()
        # CrossModalFeatureTransform
        #self.tmft = TriMFTBlock(channel, grid_size = grid_size, head_dim=head_dim)
        # CrossModalFeatureFusion
        self.tmff = TriMFFBlock(channel, m_kernel)
        self.c_rgb = channel[0]
        self.c_depth = channel[1]
        self.c_ir = channel[2]

    def forward(self, x_rgb_depth_ir):
        x_rgb, x_depth, x_ir = x_rgb_depth_ir[0], x_rgb_depth_ir[1], x_rgb_depth_ir[2]
        r_rgb, r_depth, r_ir = x_rgb, x_depth, x_ir
        fused = self.tmff(r_rgb, r_depth, r_ir)
        return fused

class TriMFTBlock(nn.Module):
    """
    空间网格 + 通道 Q/K/V 三模态融合
    把特征图分成 8×8 网格，每个网格做通道级注意力
    """

    def __init__(self, channel, grid_size=8, head_dim=64):
        super().__init__()
        c_rgb, c_depth, c_ir = channel[0], channel[1], channel[2]
        self.grid_size = grid_size
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # 每个模态的 Q/K/V（通道维度）
        self.q_rgb = nn.Linear(c_rgb, head_dim, bias=False)
        self.k_rgb = nn.Linear(c_rgb, head_dim, bias=False)
        self.v_rgb = nn.Linear(c_rgb, c_rgb, bias=False)

        self.q_depth = nn.Linear(c_depth, head_dim, bias=False)
        self.k_depth = nn.Linear(c_depth, head_dim, bias=False)
        self.v_depth = nn.Linear(c_depth, c_depth, bias=False)

        self.q_ir = nn.Linear(c_ir, head_dim, bias=False)
        self.k_ir = nn.Linear(c_ir, head_dim, bias=False)
        self.v_ir = nn.Linear(c_ir, c_ir, bias=False)

        # 跨模态投影
        self.proj_depth_to_rgb = nn.Linear(c_depth, c_rgb, bias=False)
        self.proj_ir_to_rgb = nn.Linear(c_ir, c_rgb, bias=False)
        self.proj_rgb_to_depth = nn.Linear(c_rgb, c_depth, bias=False)
        self.proj_ir_to_depth = nn.Linear(c_ir, c_depth, bias=False)
        self.proj_rgb_to_ir = nn.Linear(c_rgb, c_ir, bias=False)
        self.proj_depth_to_ir = nn.Linear(c_depth, c_ir, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        # 残差权重
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_depth = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))

    def forward(self, x_rgb, x_depth, x_ir):
        b, c_rgb, h, w = x_rgb.shape
        _, c_depth, _, _ = x_depth.shape
        _, c_ir, _, _ = x_ir.shape

        # ===== 1. 下采样到 grid_size × grid_size =====
        grid_h, grid_w = self.grid_size, self.grid_size
        rgb_grid = F.adaptive_avg_pool2d(x_rgb, (grid_h, grid_w))  # [B, c_rgb, G, G]
        depth_grid = F.adaptive_avg_pool2d(x_depth, (grid_h, grid_w))
        ir_grid = F.adaptive_avg_pool2d(x_ir, (grid_h, grid_w))

        # 展平空间维度
        rgb_grid = rgb_grid.view(b, c_rgb, -1).transpose(1, 2)  # [B, G*G, c_rgb]
        depth_grid = depth_grid.view(b, c_depth, -1).transpose(1, 2)
        ir_grid = ir_grid.view(b, c_ir, -1).transpose(1, 2)

        # ===== 2. 每个网格做 Q/K/V =====
        # RGB 的 Q/K/V
        q_rgb = self.q_rgb(rgb_grid)  # [B, L, head_dim]
        k_rgb = self.k_rgb(rgb_grid)  # [B, L, head_dim]
        v_rgb = self.v_rgb(rgb_grid)  # [B, L, c_rgb]

        q_depth = self.q_depth(depth_grid)  # [B, L, head_dim]
        k_depth = self.k_depth(depth_grid)  # [B, L, head_dim]
        v_depth = self.v_depth(depth_grid)  # [B, L, c_depth]

        q_ir = self.q_ir(ir_grid)  # [B, L, head_dim]
        k_ir = self.k_ir(ir_grid)  # [B, L, head_dim]
        v_ir = self.v_ir(ir_grid)  # [B, L, c_ir]

        # ===== 3. RGB 增强：每个网格查询 Depth 和 IR =====
        # Q @ K^T：每个网格对每个网格的注意力（空间内交互）
        attn_rgb_depth = self.softmax(q_rgb @ k_depth.transpose(1, 2) * self.scale)  # [B, L, L]
        attn_rgb_ir = self.softmax(q_rgb @ k_ir.transpose(1, 2) * self.scale)  # [B, L, L]

        # 对 L 维度做归一化
        attn_rgb_depth = attn_rgb_depth / (attn_rgb_depth.sum(dim=-1, keepdim=True) + 1e-8)
        attn_rgb_ir = attn_rgb_ir / (attn_rgb_ir.sum(dim=-1, keepdim=True) + 1e-8)

        # V 投影到 RGB 通道
        v_depth_on_rgb = self.proj_depth_to_rgb(v_depth)  # [B, L, c_rgb]
        v_ir_on_rgb = self.proj_ir_to_rgb(v_ir)  # [B, L, c_rgb]

        # 聚合
        rgb_enhanced = attn_rgb_depth @ v_depth_on_rgb + attn_rgb_ir @ v_ir_on_rgb
        # [B, L, c_rgb]

        # ===== 4. Depth 增强 =====
        attn_depth_rgb = self.softmax(q_depth @ k_rgb.transpose(1, 2) * self.scale)
        attn_depth_ir = self.softmax(q_depth @ k_ir.transpose(1, 2) * self.scale)
        attn_depth_rgb = attn_depth_rgb / (attn_depth_rgb.sum(dim=-1, keepdim=True) + 1e-8)
        attn_depth_ir = attn_depth_ir / (attn_depth_ir.sum(dim=-1, keepdim=True) + 1e-8)

        v_rgb_on_depth = self.proj_rgb_to_depth(v_rgb)
        v_ir_on_depth = self.proj_ir_to_depth(v_ir)
        depth_enhanced = attn_depth_rgb @ v_rgb_on_depth + attn_depth_ir @ v_ir_on_depth

        # ===== 5. IR 增强 =====
        attn_ir_rgb = self.softmax(q_ir @ k_rgb.transpose(1, 2) * self.scale)
        attn_ir_depth = self.softmax(q_ir @ k_depth.transpose(1, 2) * self.scale)
        attn_ir_rgb = attn_ir_rgb / (attn_ir_rgb.sum(dim=-1, keepdim=True) + 1e-8)
        attn_ir_depth = attn_ir_depth / (attn_ir_depth.sum(dim=-1, keepdim=True) + 1e-8)

        v_rgb_on_ir = self.proj_rgb_to_ir(v_rgb)
        v_depth_on_ir = self.proj_depth_to_ir(v_depth)
        ir_enhanced = attn_ir_rgb @ v_rgb_on_ir + attn_ir_depth @ v_depth_on_ir

        # ===== 6. 转回特征图 =====
        rgb_enhanced = rgb_enhanced.transpose(1, 2).view(b, c_rgb, grid_h, grid_w)
        depth_enhanced = depth_enhanced.transpose(1, 2).view(b, c_depth, grid_h, grid_w)
        ir_enhanced = ir_enhanced.transpose(1, 2).view(b, c_ir, grid_h, grid_w)

        # 上采样回原尺寸
        rgb_enhanced = F.interpolate(rgb_enhanced, size=(h, w), mode='bilinear', align_corners=False)
        depth_enhanced = F.interpolate(depth_enhanced, size=(h, w), mode='bilinear', align_corners=False)
        ir_enhanced = F.interpolate(ir_enhanced, size=(h, w), mode='bilinear', align_corners=False)

        # 加权原图
        # m_rgb = torch.sigmoid(self.gamma_rgb)
        # m_depth = torch.sigmoid(self.gamma_depth)
        # m_ir = torch.sigmoid(self.gamma_ir)

        # out_rgb = m_rgb * rgb_enhanced + (2 - m_rgb) * x_rgb
        # out_depth = m_depth * depth_enhanced + (2 - m_depth) * x_depth
        # out_ir = m_ir * ir_enhanced + (2 - m_ir) * x_ir

        out_rgb = self.gamma_rgb * rgb_enhanced + x_rgb
        out_depth = self.gamma_depth * depth_enhanced + x_depth
        out_ir = self.gamma_ir * ir_enhanced + x_ir

        return out_rgb, out_depth, out_ir

class TriMFFBlock(nn.Module):
    """RGB + Depth + IR 三模态空间 融合"""

    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        if m_kernel is None:
            m_kernel = [3, 7]

        # 每个模态用两个不同卷积核提取空间权重
        self.cv_rgb1 = Conv(channel[0], 1, m_kernel[0])
        self.cv_rgb2 = Conv(channel[0], 1, m_kernel[1])
        self.cv_depth1 = Conv(channel[1], 1, m_kernel[0])
        self.cv_depth2 = Conv(channel[1], 1, m_kernel[1])
        self.cv_ir1 = Conv(channel[2], 1, m_kernel[0])
        self.cv_ir2 = Conv(channel[2], 1, m_kernel[1])

        # 合并卷积：2通道 → 1通道
        self.merge = Conv(2, 1, 5)

        # 各模态压缩到1通道
        self.compress_rgb = Conv(channel[0], 1, 3)
        self.compress_depth = Conv(channel[1], 1, 3)
        self.compress_ir = Conv(channel[2], 1, 3)

        self.act = nn.Sigmoid()

        self.proj = Conv(channel[0] + channel[1] + channel[2], channel[0], 1)

    def forward(self, x_rgb, x_depth, x_ir):
        # 1. 各模态空间权重（多尺度融合）
        w_rgb = self.merge(torch.cat([self.cv_rgb1(x_rgb), self.cv_rgb2(x_rgb)], 1))
        w_depth = self.merge(torch.cat([self.cv_depth1(x_depth), self.cv_depth2(x_depth)], 1))
        w_ir = self.merge(torch.cat([self.cv_ir1(x_ir), self.cv_ir2(x_ir)], 1))

        glob = self.compress_rgb(x_rgb) + self.compress_depth(x_depth) + self.compress_ir(x_ir)

        # 3. 三方交叉增强：每个模态用另外两个模态的空间权重
        w_for_rgb = self.act(glob + w_depth + w_ir)  # RGB 用 Depth+IR 增强
        w_for_depth = self.act(glob + w_rgb + w_ir)  # Depth 用 RGB+IR 增强
        w_for_ir = self.act(glob + w_rgb + w_depth)  # IR 用 RGB+Depth 增强

        result_rgb = x_rgb * w_for_rgb
        result_depth = x_depth * w_for_depth
        result_ir = x_ir * w_for_ir

        fused = torch.cat([result_rgb, result_depth, result_ir], dim=1)  # [B, c_rgb+c_depth+c_ir, H, W]
        fused = self.proj(fused)
        return fused

class BiMFBlock(nn.Module):
    """三模态：RGB + Depth + IR"""

    def __init__(self, channel=(128,64), grid_size = 8, head_dim=16, m_kernel=None):
        super().__init__()
        # CrossModalFeatureTransform
        self.bmft = BiMFTBlock(channel, grid_size = grid_size, head_dim=head_dim)
        # CrossModalFeatureFusion
        self.bmff = BiMFFBlock(channel, m_kernel)
        self.c_rgb = channel[0]
        self.c_ir = channel[1]

    def forward(self, x_rgb_ir):
        x_rgb, x_ir = x_rgb_ir[0], x_rgb_ir[1]
        r_rgb, r_ir = self.bmft(x_rgb, x_ir)
        fused = self.bmff(r_rgb, r_ir)
        return fused

class BiMFTBlock(nn.Module):
    """
    空间网格 + 通道 Q/K/V 三模态融合
    把特征图分成 8×8 网格，每个网格做通道级注意力
    """

    def __init__(self, channel, grid_size=8, head_dim=64):
        super().__init__()
        c_rgb, c_ir = channel[0], channel[1]
        self.grid_size = grid_size
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # 每个模态的 Q/K/V（通道维度）
        self.q_rgb = nn.Linear(c_rgb, head_dim, bias=False)
        self.k_rgb = nn.Linear(c_rgb, head_dim, bias=False)
        self.v_rgb = nn.Linear(c_rgb, c_rgb, bias=False)

        self.q_ir = nn.Linear(c_ir, head_dim, bias=False)
        self.k_ir = nn.Linear(c_ir, head_dim, bias=False)
        self.v_ir = nn.Linear(c_ir, c_ir, bias=False)

        # 跨模态投影
        self.proj_ir_to_rgb = nn.Linear(c_ir, c_rgb, bias=False)
        self.proj_rgb_to_ir = nn.Linear(c_rgb, c_ir, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        # 残差权重
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))

    def forward(self, x_rgb, x_ir):
        b, c_rgb, h, w = x_rgb.shape
        _, c_ir, _, _ = x_ir.shape

        # ===== 1. 下采样到 grid_size × grid_size =====
        grid_h, grid_w = self.grid_size, self.grid_size
        rgb_grid = F.adaptive_avg_pool2d(x_rgb, (grid_h, grid_w))  # [B, c_rgb, G, G]
        ir_grid = F.adaptive_avg_pool2d(x_ir, (grid_h, grid_w))

        # 展平空间维度
        rgb_grid = rgb_grid.view(b, c_rgb, -1).transpose(1, 2)  # [B, G*G, c_rgb]
        ir_grid = ir_grid.view(b, c_ir, -1).transpose(1, 2)

        # ===== 2. 每个网格做 Q/K/V =====
        # RGB 的 Q/K/V
        q_rgb = self.q_rgb(rgb_grid)  # [B, L, head_dim]
        k_rgb = self.k_rgb(rgb_grid)  # [B, L, head_dim]
        v_rgb = self.v_rgb(rgb_grid)  # [B, L, c_rgb]

        q_ir = self.q_ir(ir_grid)  # [B, L, head_dim]
        k_ir = self.k_ir(ir_grid)  # [B, L, head_dim]
        v_ir = self.v_ir(ir_grid)  # [B, L, c_ir]

        # ===== 3. RGB 增强：每个网格查询 Depth 和 IR =====
        # Q @ K^T：每个网格对每个网格的注意力（空间内交互）
        attn_rgb_ir = self.softmax(q_rgb @ k_ir.transpose(1, 2) * self.scale)  # [B, L, L]
        # 对 L 维度做归一化
        attn_rgb_ir = attn_rgb_ir / (attn_rgb_ir.sum(dim=-1, keepdim=True) + 1e-8)
        # V 投影到 RGB 通道
        v_ir_on_rgb = self.proj_ir_to_rgb(v_ir)  # [B, L, c_rgb]
        # 聚合
        rgb_enhanced = attn_rgb_ir @ v_ir_on_rgb
        # [B, L, c_rgb]

        # ===== 4. Depth 增强 =====

        # ===== 5. IR 增强 =====
        attn_ir_rgb = self.softmax(q_ir @ k_rgb.transpose(1, 2) * self.scale)
        attn_ir_rgb = attn_ir_rgb / (attn_ir_rgb.sum(dim=-1, keepdim=True) + 1e-8)

        v_rgb_on_ir = self.proj_rgb_to_ir(v_rgb)
        ir_enhanced = attn_ir_rgb @ v_rgb_on_ir

        # ===== 6. 转回特征图 =====
        rgb_enhanced = rgb_enhanced.transpose(1, 2).view(b, c_rgb, grid_h, grid_w)
        ir_enhanced = ir_enhanced.transpose(1, 2).view(b, c_ir, grid_h, grid_w)

        # 上采样回原尺寸
        rgb_enhanced = F.interpolate(rgb_enhanced, size=(h, w), mode='bilinear', align_corners=False)
        ir_enhanced = F.interpolate(ir_enhanced, size=(h, w), mode='bilinear', align_corners=False)

        # 加权原图
        # m_rgb = torch.sigmoid(self.gamma_rgb)
        # m_depth = torch.sigmoid(self.gamma_depth)
        # m_ir = torch.sigmoid(self.gamma_ir)

        # out_rgb = m_rgb * rgb_enhanced + (2 - m_rgb) * x_rgb
        # out_depth = m_depth * depth_enhanced + (2 - m_depth) * x_depth
        # out_ir = m_ir * ir_enhanced + (2 - m_ir) * x_ir

        out_rgb = self.gamma_rgb * rgb_enhanced + x_rgb
        out_ir = self.gamma_ir * ir_enhanced + x_ir

        return out_rgb, out_ir

class BiMFFBlock(nn.Module):
    """RGB + Depth + IR 三模态空间 融合"""

    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        if m_kernel is None:
            m_kernel = [3, 7]

        # 每个模态用两个不同卷积核提取空间权重
        self.cv_rgb1 = Conv(channel[0], 1, m_kernel[0])
        self.cv_rgb2 = Conv(channel[0], 1, m_kernel[1])
        self.cv_ir1 = Conv(channel[1], 1, m_kernel[0])
        self.cv_ir2 = Conv(channel[1], 1, m_kernel[1])

        # 合并卷积：2通道 → 1通道
        self.merge = Conv(2, 1, 5)

        # 各模态压缩到1通道
        self.compress_rgb = Conv(channel[0], 1, 3)
        self.compress_ir = Conv(channel[1], 1, 3)

        self.act = nn.Sigmoid()

        self.proj = Conv(channel[0] + channel[1], channel[0], 1)

    def forward(self, x_rgb, x_ir):
        # 1. 各模态空间权重（多尺度融合）
        w_rgb = self.merge(torch.cat([self.cv_rgb1(x_rgb), self.cv_rgb2(x_rgb)], 1))
        w_ir = self.merge(torch.cat([self.cv_ir1(x_ir), self.cv_ir2(x_ir)], 1))

        glob = self.compress_rgb(x_rgb) + self.compress_ir(x_ir)

        # 3. 三方交叉增强：每个模态用另外两个模态的空间权重
        w_for_rgb = self.act(glob  + w_ir)  # RGB 用 Depth+IR 增强
        w_for_ir = self.act(glob + w_rgb)  # IR 用 RGB+Depth 增强

        result_rgb = x_rgb * w_for_rgb
        result_ir = x_ir * w_for_ir

        fused = torch.cat([result_rgb, result_ir], dim=1)  # [B, c_rgb+c_depth+c_ir, H, W]
        fused = self.proj(fused)
        return fused

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
        self.m = nn.ModuleList(PDCBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), mode=mode) for _ in range(n)) # identity!!!! lmc

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
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), mode: str='abcd', e: float = 0.5
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
        self.cv1 = MixedPDConv(c1, c_, k[0], 1, mode=mode)
        self.cv2 = MixedPDConv(c_, c2, k[1], 1, g=g, mode=mode)
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

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, mode='abcd'):
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
        self.default_act = nn.SiLU()  # default activation
        self.conv = PConv2d(c1, c2, k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=True, mode=mode)
        #self.conv = PConv2d(c1, c2, k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False, mode=mode)
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
                 mode='abcd'):
        super(PConv2d, self).__init__()
        conv_map = {'a': VanillaConv2d(), 'b': CDConv2d(), 'c': ADConv2d(), 'd': RDConv2d()}

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

        self.alpha_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # 三种 pdc 函数
        self.conv1 = conv_map[mode[0]]
        self.conv2 = conv_map[mode[1]]
        self.conv3 = conv_map[mode[2]]
        self.conv4 = conv_map[mode[3]]

        self.weight_cd = nn.Parameter(torch.tensor(1.0))
        self.weight_ad = nn.Parameter(torch.tensor(1.0))
        self.weight_rd = nn.Parameter(torch.tensor(1.0))

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
        out_cv = self.conv1(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # 三种差分 (共享 weight)
        out_cd = self.conv2(x_fp32, weight_fp32, bias_fp32, self.stride, self.padding, self.dilation, self.groups)
        out_ad = self.conv3(x_fp32, weight_fp32, bias_fp32, self.stride, self.padding, self.dilation, self.groups)
        out_rd = self.conv4(x_fp32, weight_fp32, bias_fp32, self.stride, self.padding, self.dilation, self.groups)

        # 三个差分的平均
        out_diff = (self.weight_cd * out_cd + self.weight_ad * out_ad + self.weight_rd * out_rd) / (
                    self.weight_cd + self.weight_ad + self.weight_rd)

        # alpha
        alpha = torch.sigmoid(self.alpha_conv(x))  # [B, C, H, W]
        out = alpha * out_cv + (1 - alpha) * out_diff

        # 转回原 dtype
        return out.to(x.dtype)

class VanillaConv2d(nn.Module):
    """普通卷积（包装 F.conv2d）"""
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)

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
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device, dtype=weight.dtype)
        w = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = w[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -w[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return F.conv2d(x, buffer, bias, stride=stride, padding=rd_padding, dilation=dilation, groups=groups)

class PDCC2fback(nn.Module):
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
        self.m = nn.ModuleList(PDCBottleneckback(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), mode=mode) for _ in range(n)) # identity!!!! lmc

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

class PDCBottleneckback(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), mode: str='abcd', e: float = 0.5
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
        self.cv1 = MixedPDConvback(c1, c_, k[0], 1, mode=mode)
        self.cv2 = MixedPDConvback(c_, c2, k[1], 1, g=g, mode=mode)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class MixedPDConvback(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, mode='abcd'):
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
        self.default_act = nn.SiLU()  # default activation
        self.conv = PConv2dback(c1, c2, k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=True, mode=mode)
        #self.conv = PConv2d(c1, c2, k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False, mode=mode)
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

class PConv2dback(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 mode='abcd'):
        super(PConv2dback, self).__init__()
        conv_map = {'a': VanillaConv2d(), 'b': CDConv2d(), 'c': ADConv2d(), 'd': RDConv2d()}

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

        self.alpha_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # 三种 pdc 函数
        self.conv1 = conv_map[mode[0]]
        self.conv2 = conv_map[mode[1]]
        self.conv3 = conv_map[mode[2]]
        self.conv4 = conv_map[mode[3]]

        self.weight_cd = nn.Parameter(torch.tensor(1.0))
        self.weight_ad = nn.Parameter(torch.tensor(1.0))
        self.weight_rd = nn.Parameter(torch.tensor(1.0))

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

        # 普通卷积
        out_cv = self.conv1(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # 三种差分 (共享 weight)
        out_cd = self.conv2(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out_ad = self.conv3(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out_rd = self.conv4(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # 三个差分的平均
        out_diff = (self.weight_cd * out_cd + self.weight_ad * out_ad + self.weight_rd * out_rd) / (
                    self.weight_cd + self.weight_ad + self.weight_rd)

        # alpha
        alpha = torch.sigmoid(self.alpha_conv(x))  # [B, C, H, W]
        out = alpha * out_cv + (1 - alpha) * out_diff

        # 转回原 dtype
        return out.to(x.dtype)
