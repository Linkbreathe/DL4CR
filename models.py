import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# --------------------------------------------------------
# Basic Building Blocks
# --------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv.

    If `bilinear=True`, use bilinear upsampling.
    Otherwise, use ConvTranspose2d.

    Note:
        `in_channels` 代表 concat 之后的通道数（skip + upsample），
        而不是 decoder 分支单独的通道数。
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: Decoder features (to be upsampled)
        x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Pad if needed (for odd spatial dims)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# --------------------------------------------------------
# Advanced Modules: scAG & NAC
# --------------------------------------------------------

class scAG(nn.Module):
    """
    Spatial-Channel Attention Gate (scAG).

    Uses decoder features (Gating Signal) to guide encoder features (Skip Connection).
    """

    def __init__(self, C_e: int, C_d: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        # 1. Projection layer: map Decoder channels (C_d) to Encoder channels (C_e)
        self.conv_query = nn.Conv2d(C_d, C_e, kernel_size=1, bias=False)
        self.bn_query = nn.BatchNorm2d(C_e)

        # 2. Channel Attention - SE style
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_e, C_e // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_e // reduction, C_e, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 3. Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                      padding=spatial_kernel // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, F_e: torch.Tensor, F_d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_e: Encoder features (Skip) [B, C_e, H, W]
            F_d: Decoder features (Gating) [B, C_d, H_d, W_d]
        """
        # --- Prepare Gating Signal ---
        gating = self.bn_query(self.conv_query(F_d))

        # Upsample gating to encoder spatial size if needed
        if gating.shape[2:] != F_e.shape[2:]:
            gating = F.interpolate(gating, size=F_e.shape[2:], mode="bilinear", align_corners=True)

        # --- Feature Fusion ---
        F_combined = F_e + gating

        # --- Channel Attention ---
        Mc = self.mlp(F_combined)  # [B, C_e, 1, 1]

        # --- Spatial Attention ---
        avg_out = torch.mean(F_combined, dim=1, keepdim=True)          # [B, 1, H, W]
        max_out, _ = torch.max(F_combined, dim=1, keepdim=True)        # [B, 1, H, W]
        Ms = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))  # [B, 1, H, W]

        # --- Final Refinement ---
        F_e_refined = F_e * Mc * Ms
        return F_e_refined


class NAC(nn.Module):
    """
    Neighbor-Aware Context Block (NAC).

    Uses convolutional branches with different dilation rates to capture multi-scale context,
    and uses input-adaptive weights to dynamically fuse these branches.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        # Four branches with different dilation rates
        self.branch1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=3, dilation=3)
        self.branch4 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=5, dilation=5)

        # Dynamic weight generator
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(in_ch, in_ch // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 4, kernel_size=1),  # 4 weights for 4 branches
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate weights [B, 4, 1, 1]
        weights = self.weight_gen(x)

        # Compute branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Weighted sum
        out = (weights[:, 0:1] * b1 +
               weights[:, 1:2] * b2 +
               weights[:, 2:3] * b3 +
               weights[:, 3:4] * b4)
        return out


# --------------------------------------------------------
# Model Definitions
# --------------------------------------------------------

class UNet2D(nn.Module):
    """
    Standard 2D U-Net.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet2D_scAG(nn.Module):
    """
    U-Net with Spatial-Channel Attention Gates (scAG).
    Applies attention mechanisms at Skip Connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # ---- Compute channel dimensions ----
        # Encoder channels
        enc_ch_1 = base_channels              # x1
        enc_ch_2 = base_channels * 2          # x2
        enc_ch_3 = base_channels * 4          # x3
        enc_ch_4 = base_channels * 8          # x4

        # Decoder gating channels:
        # x5:      base*16 // factor
        # up1 out: base*8  // factor
        # up2 out: base*4  // factor
        # up3 out: base*2  // factor
        gating_ch_4 = base_channels * 16 // factor   # for scAG1 (with x4)
        gating_ch_3 = base_channels * 8 // factor    # for scAG2 (with x3)
        gating_ch_2 = base_channels * 4 // factor    # for scAG3 (with x2)
        gating_ch_1 = base_channels * 2 // factor    # for scAG4 (with x1)

        # Attention Gates (C_e: encoder, C_d: decoder gating)
        self.scAG1 = scAG(C_e=enc_ch_4, C_d=gating_ch_4)
        self.scAG2 = scAG(C_e=enc_ch_3, C_d=gating_ch_3)
        self.scAG3 = scAG(C_e=enc_ch_2, C_d=gating_ch_2)
        self.scAG4 = scAG(C_e=enc_ch_1, C_d=gating_ch_1)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)   # [B, base,    H,   W]
        x2 = self.down1(x1)  # [B, 2*base, H/2, W/2]
        x3 = self.down2(x2)  # [B, 4*base, H/4, W/4]
        x4 = self.down3(x3)  # [B, 8*base, H/8, W/8]
        x5 = self.down4(x4)  # [B, 16*base/factor, ...]

        # Decoder + scAG
        # level 1: gating from x5
        x4_refined = self.scAG1(F_e=x4, F_d=x5)
        x = self.up1(x5, x4_refined)   # out: [B, 8*base/factor, ...]

        # level 2: gating from x (after up1)
        x3_refined = self.scAG2(F_e=x3, F_d=x)
        x = self.up2(x, x3_refined)    # out: [B, 4*base/factor, ...]

        # level 3
        x2_refined = self.scAG3(F_e=x2, F_d=x)
        x = self.up3(x, x2_refined)    # out: [B, 2*base/factor, ...]

        # level 4
        x1_refined = self.scAG4(F_e=x1, F_d=x)
        x = self.up4(x, x1_refined)

        logits = self.outc(x)
        return logits


class UNet2D_NAC(nn.Module):
    """
    U-Net with NAC Bottleneck and scAG.
    Integrates Multi-scale Context (NAC) and Attention Mechanisms (scAG).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # Bottleneck uses NAC
        bottleneck_ch = base_channels * 16 // factor
        self.bottleneck_nac = NAC(in_ch=bottleneck_ch, out_ch=bottleneck_ch)

        # ---- Compute channel dimensions ----
        # Encoder channels
        enc_ch_1 = base_channels              # x1
        enc_ch_2 = base_channels * 2          # x2
        enc_ch_3 = base_channels * 4          # x3
        enc_ch_4 = base_channels * 8          # x4

        # Decoder gating channels (same pattern as scAG-UNet, but x5 is after NAC)
        gating_ch_4 = bottleneck_ch           # x5 after NAC
        gating_ch_3 = base_channels * 8 // factor
        gating_ch_2 = base_channels * 4 // factor
        gating_ch_1 = base_channels * 2 // factor

        # Attention Gates
        self.scAG1 = scAG(C_e=enc_ch_4, C_d=gating_ch_4)
        self.scAG2 = scAG(C_e=enc_ch_3, C_d=gating_ch_3)
        self.scAG3 = scAG(C_e=enc_ch_2, C_d=gating_ch_2)
        self.scAG4 = scAG(C_e=enc_ch_1, C_d=gating_ch_1)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # NAC bottleneck
        x5 = self.bottleneck_nac(x5)

        # Decoder + scAG
        x4_refined = self.scAG1(F_e=x4, F_d=x5)
        x = self.up1(x5, x4_refined)

        x3_refined = self.scAG2(F_e=x3, F_d=x)
        x = self.up2(x, x3_refined)

        x2_refined = self.scAG3(F_e=x2, F_d=x)
        x = self.up3(x, x2_refined)

        x1_refined = self.scAG4(F_e=x1, F_d=x)
        x = self.up4(x, x1_refined)

        logits = self.outc(x)
        return logits
