import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Dict, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================
# 1. 基本配置 & 工具函数
# =============================

IMG_HEIGHT = 224   # 跟训练脚本保持一致
IMG_WIDTH = 320
MEAN = (0.5,)      # 灰度归一化均值
STD = (0.25,)      # 灰度归一化方差


def set_seed(seed: int = 42) -> None:
    """固定随机种子，确保 split / 评估可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_grayscale_image(path: Path) -> np.ndarray:
    """
    从磁盘读取一张灰度图，返回 numpy 数组 (H, W)，范围 0~255，dtype=uint8。
    用于可视化时展示“原始外观”（不做归一化）。
    """
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path).convert("L")
    return np.array(img)


# =============================
# 2. Albumentations 测试集变换
# =============================

def get_test_transforms() -> A.Compose:
    """
    测试 / 验证阶段的图像变换。
    注意：**不再使用 Crop_noise**，只做 Resize + Normalize + ToTensorV2，
    与你目前的 fine-tune 训练脚本保持完全一致。
    """
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# =============================
# 3. 数据集定义（与训练版一致，无 Crop）
# =============================

class ThyroidDataset(Dataset):
    """
    Thyroid 分割数据集。

    - image_ids: 文件名列表（不含路径，可能含扩展名也可能不含）
    - images_dir: 图像所在目录
    - masks_dir : 掩膜所在目录
    - transform : Albumentations 变换（用于 image & mask）
    """

    def __init__(
        self,
        image_ids: List[str],
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None,
    ):
        self.image_ids = image_ids
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img_path = self._find_file(self.images_dir, img_id)
        mask_path = self._find_file(self.masks_dir, img_id)

        # 读成灰度 numpy
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Albumentations 变换（保证 image & mask 同样的几何变化）
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # Tensor
            mask = augmented["mask"]     # Tensor

        # 掩膜二值化到 {0,1}
        if isinstance(mask, torch.Tensor):
            mask = mask.float() / 255.0 if mask.max() > 1.0 else mask.float()
            mask = (mask > 0.5).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)
            mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        # 返回：变换后的图像 / 掩膜 / 原始图像路径
        return image, mask, str(img_path)

    def _find_file(self, folder: Path, filename: str) -> Path:
        """
        根据给定 ID 在目录中查找对应图像文件。
        - 如果 filename 本身有扩展名，直接尝试；
        - 否则自动补全常见扩展名。
        """
        # 1. 直接用 filename
        direct = folder / filename
        if direct.exists():
            return direct

        # 2. 尝试常见扩展名
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            full = folder / (filename + ext)
            if full.exists():
                return full

        raise FileNotFoundError(f"Could not find file for ID {filename} in {folder}")


# =============================
# 4. 模型定义（与 fine-tune 脚本一致）
# =============================

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
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

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样：MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样：Upsample/ConvTranspose -> 拼接 skip -> DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 来自上一层 decoder
        # x2: 来自 encoder 的 skip connection
        x1 = self.up(x1)

        # 处理由于卷积/池化造成的尺寸差异（pad 一下）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """最后的 1x1 conv——输出通道数 = n_classes"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    与你 fine-tune 脚本一致的 U-Net：
    - base_channels 控制容量（默认为 32）
    - bilinear 控制上采样方式
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
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


# =============================
# 5. 损失函数（与 fine-tune 一致）
# =============================

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss（从 logits 直接计算）：
        Dice = 2 * (p ∩ y) / (p + y)
        DiceLoss = 1 - Dice

    - logits: (B, 1, H, W)
    - targets: (B, 1, H, W)，元素在 {0,1}
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def combined_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lambda_dice: float = 0.2,
) -> torch.Tensor:
    """
    组合分割损失（与你同学一致的设计思想）：
        loss = BCEWithLogitsLoss + lambda_dice * DiceLoss
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss_from_logits(logits, targets)
    return bce + lambda_dice * dloss


# =============================
# 6. 指标计算（按你同学的方式）
# =============================

def compute_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Dice 计算方式与同学一致：
    1. logits -> sigmoid -> 二值化（阈值=0.5）
    2. 每张图 flatten 成向量
    3. 对每张图单独算 Dice，再对 batch 取平均
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)   # (B,)
    return dice.mean().item()


def compute_iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    IoU 计算方式：同样是 per-sample 再在 batch 内平均。
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)          # (B,)
    return iou.mean().item()


def compute_precision_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> (float, float):
    """
    Precision & Recall 计算方式：
    - 对每张图单独算 TP / (TP+FP), TP / (TP+FN)，再在 batch 内平均。
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)   # TP
    pred_sum = preds_flat.sum(dim=1)                        # TP + FP
    target_sum = targets_flat.sum(dim=1)                    # TP + FN

    precision = (intersection + eps) / (pred_sum + eps)     # (B,)
    recall = (intersection + eps) / (target_sum + eps)      # (B,)

    return precision.mean().item(), recall.mean().item()


# =============================
# 7. 可视化函数（按你同学的风格）
# =============================

def plot_sample(
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    title: str,
    save_path: Path,
):
    """
    绘制四联图：
        1. 原始 X-ray（灰度）
        2. GT 掩膜
        3. 预测掩膜
        4. 原图 + 预测掩膜红色覆盖的 overlay
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)

    # 1) 原始图
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input Ultrasound")
    axes[0].axis("off")

    # 2) GT Mask
    axes[1].imshow(gt_mask_np, cmap="gray")
    axes[1].set_title("Ground Truth mask")
    axes[1].axis("off")

    # 3) Pred Mask
    axes[2].imshow(pred_mask_np, cmap="gray")
    axes[2].set_title("Predicted mask")
    axes[2].axis("off")

    # 4) Overlay：在原图上用红色显示预测前景
    # 将灰度图扩展成 RGB
    overlay_img = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    red_mask = np.zeros_like(overlay_img, dtype=np.uint8)
    red_mask[pred_mask_np == 1] = [255, 0, 0]  # 红色前景

    final_overlay = cv2.addWeighted(overlay_img, 1.0, red_mask, 0.5, 0)

    axes[3].imshow(final_overlay)
    axes[3].set_title("Overlay (prediction)")
    axes[3].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)


def visualize_samples(
    model: nn.Module,
    device: torch.device,
    test_ds: Dataset,
    sample_metas: List[Dict[str, Any]],
    all_sample_losses: List[float],
    title_prefix: str,
    save_dir: Path,
    lambda_dice: float = 0.2,
):
    """
    按你同学的风格：
        - 输入若干 sample_metas（里面包含 index & img_path）
        - 对每个样本：
            1. 用原始图片（只做 resize）当背景
            2. 用 test_ds 提供的变换后图像喂模型，得到预测 mask
            3. 调用 plot_sample 画出四联图并保存
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, meta in enumerate(sample_metas):
        idx = meta["index"]
        img_path = Path(meta["img_path"])

        # 1. 加载原始灰度图，并 resize 到 (IMG_HEIGHT, IMG_WIDTH)，便于和 mask 对齐
        original_img_np = load_grayscale_image(img_path)
        resized_original = cv2.resize(
            original_img_np, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

        # 2. 从 test_ds 中获取变换后的图像 & 掩膜，用于喂模型和可视化
        img_t, mask_t, _ = test_ds[idx]
        img_t = img_t.unsqueeze(0).to(device)     # (1, 1, H, W)
        mask_t = mask_t.to(device).unsqueeze(0) if mask_t.ndim == 2 else mask_t.to(device)  # 容错处理

        with torch.no_grad():
            logits = model(img_t)
            probs = torch.sigmoid(logits)
            pred_mask_t = (probs > 0.5).float()   # 二值预测

        # 转成 numpy 用于画图
        pred_mask_np = pred_mask_t.squeeze().cpu().numpy().astype(np.uint8)
        gt_mask_np = mask_t.squeeze().cpu().numpy().astype(np.uint8)

        # 当前样本的 loss（已在外面算过）
        current_loss = all_sample_losses[idx]

        title = f"{title_prefix} Sample {i+1} (idx={idx}, Loss={current_loss:.4f})"
        save_path = save_dir / f"{title_prefix.lower().replace(' ', '_')}_{i+1}_idx{idx}.png"

        plot_sample(resized_original, gt_mask_np, pred_mask_np, title, save_path)


# =============================
# 8. 主评估逻辑
# =============================

def evaluate_model(args):
    # ---- 设备 & 随机种子 ----
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Running evaluation on {device}...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 构造数据划分（必须与训练脚本保持一致） ----
    all_image_files = sorted(os.listdir(args.images_dir))
    all_image_files = [
        f for f in all_image_files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if args.split_json:
        print(f"Loading split from {args.split_json}...")
        with open(args.split_json, "r") as f:
            split_data = json.load(f)

        def get_files_from_ids(id_list, available_files):
            file_map = {os.path.splitext(f)[0]: f for f in available_files}
            res = []
            for i in id_list:
                i = str(i)
                base = os.path.splitext(i)[0]
                if base in file_map:
                    res.append(file_map[base])
            return res

        # 训练脚本里：train 来自 split_data['train']，
        # val_all_files 来自 split_data['val']，再 50/50 划分为 val/test
        train_files = get_files_from_ids(split_data["train"], all_image_files)
        val_all_files = get_files_from_ids(split_data["val"], all_image_files)

        random.seed(args.seed)
        random.shuffle(val_all_files)
        split_point = len(val_all_files) // 2
        val_files = val_all_files[:split_point]
        test_files = val_all_files[split_point:]

        print(f"Data split (reconstructed): Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    else:
        # 无 JSON 时，按随机比例划分（与训练脚本一致）
        print("No split JSON provided. Using random split.")
        random.seed(args.seed)
        random.shuffle(all_image_files)
        val_count = int(len(all_image_files) * args.val_ratio)
        test_count = int(len(all_image_files) * args.test_ratio)
        train_count = len(all_image_files) - val_count - test_count

        train_files = all_image_files[:train_count]
        val_files = all_image_files[train_count:train_count + val_count]
        test_files = all_image_files[train_count + val_count:]

        print(f"Data split (random): Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # ---- 2. Dataset & DataLoader ----
    test_ds = ThyroidDataset(
        test_files,
        args.images_dir,
        args.masks_dir,
        transform=get_test_transforms(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Evaluated Test Set Size: {len(test_ds)}")

    # ---- 3. 构建模型并加载 checkpoint ----
    model = UNet(
        n_channels=1,
        n_classes=1,
        base_channels=args.base_channels,
        bilinear=not args.no_bilinear,
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 支持不同保存格式：{'model_state_dict': ...} / {'state_dict': ...} / {'model': ...} / 纯 state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Detected 'model_state_dict' key, unpacking...")
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Detected 'state_dict' key, unpacking...")
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        print("Detected 'model' key, unpacking...")
        model.load_state_dict(checkpoint["model"])
    else:
        print("Loading checkpoint directly as state_dict...")
        model.load_state_dict(checkpoint)

    model.eval()

    # ---- 4. 主评估循环（按你同学的 Dice / IoU 实现） ----
    print("Starting evaluation on test set...")

    test_loss_sum = 0.0
    test_batches = 0

    dice_sum = 0.0
    iou_sum = 0.0
    prec_sum = 0.0
    recall_sum = 0.0

    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            logits = model(images)
            loss = combined_segmentation_loss(logits, masks, lambda_dice=args.lambda_dice)

            test_loss_sum += loss.item()
            test_batches += 1

            batch_dice = compute_dice_score(logits, masks)
            batch_iou = compute_iou_score(logits, masks)
            batch_prec, batch_recall = compute_precision_recall(logits, masks)

            dice_sum += batch_dice
            iou_sum += batch_iou
            prec_sum += batch_prec
            recall_sum += batch_recall

    avg_test_loss = test_loss_sum / max(1, test_batches)
    avg_dice = dice_sum / max(1, test_batches)
    avg_iou = iou_sum / max(1, test_batches)
    avg_prec = prec_sum / max(1, test_batches)
    avg_recall = recall_sum / max(1, test_batches)

    # ---- 5. 打印总体评估结果 ----
    print("\n" + "=" * 30)
    print("      EVALUATION RESULTS      ")
    print("=" * 30)
    print(f"Model: {Path(args.checkpoint).name}")
    print(f"Test Samples: {len(test_ds)}")
    print("-" * 30)
    print(f"Test Loss : {avg_test_loss:.4f}")
    print(f"Dice Score: {avg_dice:.4f}")
    print(f"IoU Score : {avg_iou:.4f}")
    print(f"Precision : {avg_prec:.4f}")
    print(f"Recall    : {avg_recall:.4f}")
    print("=" * 30)

    # ---- 6. 逐样本 loss，用于挑选 best/worst ----
    print("\nComputing per-sample losses for best/worst visualization...")

    all_sample_losses: List[float] = []
    all_sample_metas: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_t, mask_t, img_path = test_ds[idx]
            img_t = img_t.unsqueeze(0).to(device)   # (1, 1, H, W)
            mask_t = mask_t.unsqueeze(0).to(device) # (1, 1, H, W)

            logits = model(img_t)
            loss = combined_segmentation_loss(logits, mask_t, lambda_dice=args.lambda_dice)

            all_sample_losses.append(loss.item())
            all_sample_metas.append({
                "index": idx,
                "img_path": img_path,
            })

    # 排序：loss 越小表现越好
    sorted_indices = np.argsort(all_sample_losses)
    top_best_indices = sorted_indices[:args.vis_count]               # 最好的若干个
    top_worst_indices = sorted_indices[-args.vis_count:][::-1]       # 最差的若干个（从大到小）

    best_sample_metas = [all_sample_metas[i] for i in top_best_indices]
    worst_sample_metas = [all_sample_metas[i] for i in top_worst_indices]

    print(f"\nTop {args.vis_count} best performing samples (lowest loss):")
    for i, idx in enumerate(top_best_indices):
        print(f"  #{i+1}: idx={idx}, loss={all_sample_losses[idx]:.4f}, path={all_sample_metas[idx]['img_path']}")

    print(f"\nTop {args.vis_count} worst performing samples (highest loss):")
    for i, idx in enumerate(top_worst_indices):
        print(f"  #{i+1}: idx={idx}, loss={all_sample_losses[idx]:.4f}, path={all_sample_metas[idx]['img_path']}")

    # ---- 7. 可视化 best / worst 样本 ----
    print("\nVisualizing best performing samples...")
    visualize_samples(
        model=model,
        device=device,
        test_ds=test_ds,
        sample_metas=best_sample_metas,
        all_sample_losses=all_sample_losses,
        title_prefix="Best Performing",
        save_dir=vis_dir / "best",
        lambda_dice=args.lambda_dice,
    )

    print("Visualizing worst performing samples...")
    visualize_samples(
        model=model,
        device=device,
        test_ds=test_ds,
        sample_metas=worst_sample_metas,
        all_sample_losses=all_sample_losses,
        title_prefix="Worst Performing",
        save_dir=vis_dir / "worst",
        lambda_dice=args.lambda_dice,
    )

    print(f"\nAll visualizations saved under: {vis_dir}")


# =============================
# 9. 参数解析
# =============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TG3K segmentation model (metrics & visualization, aligned with teammate's style)"
    )

    # 路径相关
    parser.add_argument("--images-dir", type=str, required=True, help="Directory of input images")
    parser.add_argument("--masks-dir", type=str, required=True, help="Directory of GT masks")
    parser.add_argument("--split-json", type=str, default=None, help="Path to tg3k-trainval.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to segmentation checkpoint (best_model.pth)")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_results"), help="Directory to save results")

    # 模型结构，与训练对齐
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base number of feature channels in U-Net")
    parser.add_argument("--no-bilinear", action="store_true",
                        help="Use ConvTranspose instead of bilinear upsampling")

    # 评估设置
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=0, help="Num DataLoader workers")
    parser.add_argument("--lambda-dice", type=float, default=0.2,
                        help="Weight for Dice loss term in combined loss (for per-sample loss ranking)")
    parser.add_argument("--vis-count", type=int, default=5,
                        help="Number of best/worst samples to visualize each")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match training for same split)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # 当没有 JSON 时用于随机划分（保持与训练逻辑同构）
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)


# python tg3k_visualize_results.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --checkpoint ./tg3k/checkpoints_finetune/mim/best_model.pth --output-dir ./tg3k/eval_results/mim_final_test --seed 42