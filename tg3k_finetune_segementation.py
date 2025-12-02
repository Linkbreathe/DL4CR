import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ============================================================
# 1. 通用工具：随机种子
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. 三段式 transforms（common / image / mask），不含 Crop_noise
# ============================================================

TARGET_HEIGHT = 224
TARGET_WIDTH = 320
MEAN = (0.5,)   # 单通道灰度
STD = (0.25,)


def get_segmentation_train_transforms_split():
    """
    Train 阶段的三段式 transforms：

    - common_transforms: 同时作用在 image 和 mask 上的几何变换
        * Resize
        * HorizontalFlip
        * Rotate
    - image_transforms: 只作用在 image 的像素级操作
        * Normalize(mean=0.5, std=0.25)
        * ToTensorV2
    - mask_transforms: 只作用在 mask 的操作
        * ToFloat(max_value=255.0)
        * ToTensorV2
    """
    common_transforms = A.Compose(
        [
            A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        ],
        additional_targets={"mask": "mask"},
    )

    image_transforms = A.Compose(
        [
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )

    mask_transforms = A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )

    return common_transforms, image_transforms, mask_transforms


def get_segmentation_val_transforms_split():
    """
    Val/Test 阶段的三段式 transforms：

    - common_transforms: 只做确定性的 Resize（不做 Flip/Rotate）
    - image_transforms: Normalize + ToTensor
    - mask_transforms: ToFloat + ToTensor
    """
    common_transforms = A.Compose(
        [
            A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
        ],
        additional_targets={"mask": "mask"},
    )

    image_transforms = A.Compose(
        [
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )

    mask_transforms = A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )

    return common_transforms, image_transforms, mask_transforms


# ============================================================
# 3. 数据集：三段式 ThyroidDataset
# ============================================================

class ThyroidDataset(Dataset):
    """
    三段式数据集实现：
      - image_ids: 文件名列表（可以带扩展名或只是 stem）
      - images_dir: 图像目录
      - masks_dir: 掩码目录
      - common_transforms: 同时对 image / mask 做的几何变换
      - image_transforms: 只对 image 做的变换（Normalize + ToTensor）
      - mask_transforms: 只对 mask 做的变换（ToFloat + ToTensor + 后续二值化）
    """

    def __init__(
        self,
        image_ids: List[str],
        images_dir: str,
        masks_dir: str,
        common_transforms: A.Compose,
        image_transforms: A.Compose,
        mask_transforms: A.Compose,
    ):
        self.image_ids = list(image_ids)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.common_transforms = common_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]

        img_path = self._find_file(self.images_dir, img_id)
        mask_path = self._find_file(self.masks_dir, img_id)

        # 以灰度方式读入
        img_np = np.array(Image.open(img_path).convert("L"))   # (H, W)
        img_np = np.expand_dims(img_np, axis=-1)               # (H, W, 1) for albumentations
        mask_np = np.array(Image.open(mask_path).convert("L")) # (H, W)

        # 1) 公共几何变换：保证 image / mask 对齐
        augmented_common = self.common_transforms(image=img_np, mask=mask_np)
        img_common = augmented_common["image"]   # (H, W, 1) 或 (H, W)
        mask_common = augmented_common["mask"]   # (H, W)

        # 2) image-only 变换：Normalize + ToTensorV2
        img_t = self.image_transforms(image=img_common)["image"]  # Tensor (1, H, W)

        # 3) mask-only 变换：ToFloat + ToTensorV2
        mask_t = self.mask_transforms(image=mask_common)["image"]  # Tensor (H, W) 或 (1, H, W)

        # 保证 mask 有 channel 维度
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)  # (H, W) -> (1, H, W)

        # 二值化到 {0,1} —— 在 ToFloat 之后做
        mask_t = (mask_t > 0.5).float()

        meta = str(img_path)
        return img_t, mask_t, meta

    def _find_file(self, folder: Path, filename: str) -> Path:
        """
        根据文件名（可带/不带扩展名）在 folder 下自动尝试常见扩展名。
        """
        candidate = folder / filename
        if candidate.exists():
            return candidate

        stem = Path(filename).stem
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            full = folder / (stem + ext)
            if full.exists():
                return full

        raise FileNotFoundError(f"Could not find file for ID {filename} in {folder}")


# ============================================================
# 4. 模型结构：UNet2D（与 MIM 预训练脚本完全一致）
# ============================================================

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
    """下采样：MaxPool2d -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样：Upsample 或 ConvTranspose2d -> 拼接 -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: decoder 上一层输出
        # x2: encoder 对应层（skip 连接）
        x1 = self.up(x1)

        # 处理尺寸对不齐（奇数尺寸）
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
    """最后 1x1 卷积到输出通道"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    """
    与 MIM 预训练脚本中的 UNet2D 完全一致的结构：
      - in_channels = 1
      - out_channels = 1
      - base_channels 默认 32
      - bilinear 控制是否使用双线性上采样
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


# ============================================================
# 5. 损失函数 & 指标（对齐 baseline：BCE + λ·Dice）
# ============================================================

def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
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
    组合损失：BCEWithLogits + λ * DiceLoss
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss_from_logits(logits, targets)
    return bce + lambda_dice * dloss


def compute_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


# ============================================================
# 6. 从 MIM 预训练 checkpoint 加载权重
# ============================================================

def load_pretrained_weights(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """
    从 MIM 预训练的 checkpoint 中加载权重。

    支持：
      - {'model_state_dict': ...}
      - {'state_dict': ...}
      - {'model': ...}
      - 纯 state_dict

    只加载 name + shape 都匹配的层，其余保持随机初始化。
    """
    print(f"Loading pretrained weights from: {path}")
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and hasattr(v, "shape") and v.shape == model_dict[k].shape
    }
    missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())

    if not pretrained_dict:
        print("WARNING: no matching keys found between pretrained checkpoint and model.")
    else:
        print(f"Loaded {len(pretrained_dict)} layers from pretrained checkpoint.")
        if missing_keys:
            print(f"{len(missing_keys)} layers remain randomly initialized "
                  f"(e.g. {list(missing_keys)[:5]}...)")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


# ============================================================
# 7. 评估函数（val/test 共用，返回 mean Dice）
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dice_sum = 0.0
    steps = 0

    for images, masks, _ in loader:
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.float32)

        logits = model(images)
        dice = compute_dice_score(logits, masks)
        dice_sum += dice
        steps += 1

    return dice_sum / steps if steps > 0 else 0.0


# ============================================================
# 8. 训练 + Early Stopping + 使用 best_model 做 Test 评估
# ============================================================

def train_and_evaluate(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # -----------------------------
    # 数据文件收集
    # -----------------------------
    all_image_files = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    )

    # -----------------------------
    # 数据划分：优先使用 split-json
    # -----------------------------
    if args.split_json:
        print(f"Loading split from {args.split_json}...")
        with open(args.split_json, 'r') as f:
            split_data = json.load(f)

        def get_files_from_ids(id_list, available_files):
            file_map = {os.path.splitext(f)[0]: f for f in available_files}
            res = []
            for i in id_list:
                try:
                    base = f"{int(i):04d}"
                except Exception:
                    base = str(i)
                if base in file_map:
                    res.append(file_map[base])
            return res

        train_files = get_files_from_ids(split_data['train'], all_image_files)
        val_all_files = get_files_from_ids(split_data['val'], all_image_files)

        # 可选：子采样 train 到固定数量
        if args.train_subset_size > 0 and len(train_files) > args.train_subset_size:
            print(f"Subsampling training set from {len(train_files)} to "
                  f"{args.train_subset_size} (seed={args.seed}).")
            random.seed(args.seed)
            train_files = random.sample(train_files, args.train_subset_size)

        random.seed(args.seed)
        random.shuffle(val_all_files)
        split_point = len(val_all_files) // 2
        val_files = val_all_files[:split_point]
        test_files = val_all_files[split_point:]

        print(f"Data split from JSON: Train={len(train_files)}, "
              f"Val={len(val_files)}, Test={len(test_files)}")
    else:
        print("No split JSON provided. Using random split.")
        random.seed(args.seed)
        random.shuffle(all_image_files)
        val_count = int(len(all_image_files) * args.val_ratio)
        test_count = int(len(all_image_files) * args.test_ratio)
        train_count = len(all_image_files) - val_count - test_count

        train_files = all_image_files[:train_count]
        val_files = all_image_files[train_count:train_count + val_count]
        test_files = all_image_files[train_count + val_count:]

        print(f"Random Split: Train={len(train_files)}, "
              f"Val={len(val_files)}, Test={len(test_files)}")

    # -----------------------------
    # 构建三段式 Dataset & DataLoader
    # -----------------------------
    train_common, train_img_tf, train_mask_tf = get_segmentation_train_transforms_split()
    val_common, val_img_tf, val_mask_tf = get_segmentation_val_transforms_split()

    train_ds = ThyroidDataset(
        image_ids=train_files,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        common_transforms=train_common,
        image_transforms=train_img_tf,
        mask_transforms=train_mask_tf,
    )
    val_ds = ThyroidDataset(
        image_ids=val_files,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        common_transforms=val_common,
        image_transforms=val_img_tf,
        mask_transforms=val_mask_tf,
    )
    test_ds = ThyroidDataset(
        image_ids=test_files,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        common_transforms=val_common,   # test 也用 val 的（无随机增强）
        image_transforms=val_img_tf,
        mask_transforms=val_mask_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------------
    # 初始化模型（与 MIM 预训练 UNet2D 完全同构）
    # -----------------------------
    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        bilinear=not args.no_bilinear,
    ).to(device)
    print(f"UNet2D initialized with base_channels={args.base_channels}, "
          f"bilinear={not args.no_bilinear}")

    # 预训练权重（MIM）
    if args.init_from:
        model = load_pretrained_weights(model, args.init_from, device)

    # 如果需要从完整 checkpoint 恢复
    if args.resume:
        print(f"Resuming training from full model checkpoint: {args.resume}...")
        resume_ckpt = torch.load(args.resume, map_location=device)
        if isinstance(resume_ckpt, dict) and "state_dict" in resume_ckpt:
            model.load_state_dict(resume_ckpt["state_dict"])
        else:
            model.load_state_dict(resume_ckpt)

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # -----------------------------
    # TensorBoard
    # -----------------------------
    if args.tensorboard_dir:
        log_dir = Path(args.tensorboard_dir)
    else:
        log_dir = Path(args.save_dir) / "runs"

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to {log_dir}")

    # -----------------------------
    # 训练循环 + 早停
    # -----------------------------
    best_val_dice = 0.0
    epochs_without_improvement = 0
    global_step = 0

    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs} [train]", unit='batch') as pbar:
            for images, masks, _ in train_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)

                logits = model(images)
                loss = combined_segmentation_loss(logits, masks, lambda_dice=args.lambda_dice)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                writer.add_scalar('train/loss_step', loss.item(), global_step)

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        avg_train_loss = epoch_loss / max(1, num_batches)
        writer.add_scalar('train/loss_epoch', avg_train_loss, epoch)

        # ---- Validate (Dice) ----
        val_dice = evaluate(model, val_loader, device)
        writer.add_scalar('val/dice', val_dice, epoch)

        print(
            f"Epoch {epoch}/{args.epochs} finished. "
            f"Train Loss: {avg_train_loss:.4f}, Val Dice: {val_dice:.4f}"
        )

        # ---- 保存 best model & 早停 ----
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved to {best_path} (Val Dice: {best_val_dice:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement in Dice for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    writer.close()
    print("Training finished.")

    # -----------------------------
    # 使用 best_model 在 Test set 上评估 Dice
    # -----------------------------
    if best_path.exists():
        print(f"Loading best model from {best_path} for final test evaluation...")
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print("Best model file not found, using last epoch model for test evaluation.")

    test_dice = evaluate(model, test_loader, device)
    print(f"Final Test Dice: {test_dice:.4f}")


# ============================================================
# 9. 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune UNet2D segmentation on TG3K with MIM-pretrained weights "
                    "and 3-stage transforms (common/image/mask)."
    )

    # 路径
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="Directory containing masks")
    parser.add_argument("--split-json", type=str, default=None,
                        help="Path to tg3k-trainval.json (recommended)")
    parser.add_argument("--save-dir", type=str,
                        default="./tg3k/checkpoints_finetune",
                        help="Directory to save checkpoints")
    parser.add_argument("--tensorboard-dir", type=str, default=None,
                        help="Directory for TensorBoard logs (optional)")

    # 预训练 & 恢复
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to pretraining checkpoint (e.g., MIM mim_best.pth)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to full model checkpoint to resume from")

    # 模型超参数（需与 MIM 预训练一致）
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base number of feature maps in UNet2D (match MIM pretrain)")
    parser.add_argument("--no-bilinear", action="store_true",
                        help="Use ConvTranspose instead of bilinear upsampling")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default 4 matches baseline)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for Adam, e.g. 1e-5")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    # BCE + Dice 中的 λ
    parser.add_argument("--lambda-dice", type=float, default=0.2,
                        help="Weight for Dice loss term in combined loss (BCE + λ*Dice)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience based on validation Dice")

    # 没有 split_json 时的随机划分比例
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    # 使用 split_json 时，可选的训练子集大小（对齐 baseline 500 张）
    parser.add_argument("--train-subset-size", type=int, default=0,
                        help="If >0 and using split-json, subsample this many training images")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
