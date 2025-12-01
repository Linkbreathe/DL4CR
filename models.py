# -----------------------------
# Model: 2D U-Net (shared with pretraining)
# -----------------------------
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Upscaling then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

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


class UNet2D(nn.Module):
    """Standard 2D U-Net for binary segmentation (1 logit channel)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, bilinear: bool = True):
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

class scAG(nn.Module):
    def __init__(self, C_e, C_d, reduction=16, spatial_kernel=7):
        super().__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_e, C_e // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_e // reduction, C_e, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F_e, F_d):
        # Channel attention
        Mc = self.mlp(F_e)  # [B,C,1,1]
        # Spatial attention
        avg = torch.mean(F_e, dim=1, keepdim=True)
        max_ = torch.max(F_e, dim=1, keepdim=True)[0]
        Ms = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        # Refined encoder features
        F_e_refined = F_e * Mc * Ms
        return F_e_refined

class UNet2D_scAG(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, bilinear: bool = True):
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

        self.scAG1 = scAG(C_e=base_channels * 8, C_d=base_channels * 8 // factor)
        self.scAG2 = scAG(C_e=base_channels * 4, C_d=base_channels * 4 // factor)
        self.scAG3 = scAG(C_e=base_channels * 2, C_d=base_channels * 2 // factor)
        self.scAG4 = scAG(C_e=base_channels, C_d=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_refined = self.scAG1(x4, x5)
        x = self.up1(x5, x4_refined)
        x3_refined = self.scAG2(x3, x)
        x = self.up2(x, x3_refined)
        x2_refined = self.scAG3(x2, x)
        x = self.up3(x, x2_refined)
        x1_refined = self.scAG4(x1, x)
        x = self.up4(x, x1_refined)
        logits = self.outc(x)
        return logits

class NAC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Four dilated conv branches
        self.branch1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=3, dilation=3)
        self.branch4 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=5, dilation=5)

        # Adaptive weight generator (input-dependent)
        # Global pooling -> small MLP -> 4 weights
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # [B, C, 1, 1]
            nn.Conv2d(in_ch, in_ch//4, 1), # reduce
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//4, 4, 1),     # generate 4 branch weights
            nn.Softmax(dim=1)              # normalize across 4 weights
        )

    def forward(self, x):
        # Dynamic weights: shape [B, 4, 1, 1]
        weights = self.weight_gen(x)

        # Compute branch outputs
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # Weighted sum (branch axis = channel 1 of weights)
        out = (weights[:, 0:1] * b1 +
               weights[:, 1:2] * b2 +
               weights[:, 2:3] * b3 +
                weights[:, 3:4] * b4)

        return out

class UNet2D_NAC(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, bilinear: bool = True):
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

        self.bottleneck_nac = NAC(in_ch=base_channels * 16 // factor, out_ch=base_channels * 16 // factor)
        self.scAG1 = scAG(C_e=base_channels * 8, C_d=base_channels * 8 // factor)
        self.scAG2 = scAG(C_e=base_channels * 4, C_d=base_channels * 4 // factor)
        self.scAG3 = scAG(C_e=base_channels * 2, C_d=base_channels * 2 // factor)
        self.scAG4 = scAG(C_e=base_channels, C_d=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #bottleneck layer
        x5 = self.bottleneck_nac(x5)
        x4_refined = self.scAG1(x4, x5)
        x = self.up1(x5, x4_refined)
        x3_refined = self.scAG2(x3, x)
        x = self.up2(x, x3_refined)
        x2_refined = self.scAG3(x2, x)
        x = self.up3(x, x2_refined)
        x1_refined = self.scAG4(x1, x)
        x = self.up4(x, x1_refined)
        logits = self.outc(x)
        return logits
