import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


# ============================================================
# 1. General Utility: Random Seed
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Transforms (aligned with finetune script: no Crop_noise)
# ============================================================

def get_transforms(stage: str):
    """
    Data augmentation pipeline used in finetune stage.
      - train: Resize + Flip + Rotate + Normalize + ToTensor
      - valid/test: Resize + Normalize + ToTensor
    This evaluation script only uses the valid/test pipeline.
    """
    IMG_HEIGHT = 224
    IMG_WIDTH = 320
    MEAN = (0.5,)
    STD = (0.25,)

    if stage == "train":
        return A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    elif stage in ["valid", "test"]:
        return A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown stage: {stage}")


# ============================================================
# 3. Dataset (aligned with ThyroidDataset in finetune script)
# ============================================================

class ThyroidDataset(Dataset):
    """
    Simplified dataset implementation:
      - image_ids: list of filenames (e.g., "0001.png" or "0001")
      - images_dir: directory of thyroid ultrasound images
      - masks_dir: directory of segmentation masks
      - transform: Albumentations augmentation pipeline
    """

    def __init__(
        self,
        image_ids: List[str],
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None,
    ):
        self.image_ids = list(image_ids)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]

        # Automatically resolve file extension
        img_path = self._find_file(self.images_dir, img_id)
        mask_path = self._find_file(self.masks_dir, img_id)

        # Load as grayscale (H,W) → expand to (H,W,1)
        image = np.array(Image.open(img_path).convert("L"))
        image = np.expand_dims(image, axis=-1)

        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # Tensor (1,H,W)
            mask = augmented["mask"]     # Tensor (H,W) or (1,H,W)

        # Binarize mask to {0,1}
        if isinstance(mask, torch.Tensor):
            mask = mask.float()
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = (mask > 0.5).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)
            mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        # meta stores full path like your classmate’s implementation
        meta = str(img_path)

        return image, mask, meta

    def _find_file(self, folder: Path, filename: str) -> Path:
        """
        Resolve file path by trying common image extensions if extension is missing.
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
# 4. UNet2D Model (exactly same structure as pretrain/finetune)
# ============================================================

class DoubleConv(nn.Module):
    """Two consecutive blocks of (Conv2d → BN → ReLU)."""

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
    """Downscaling: MaxPool2d → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling with either bilinear upsampling or ConvTranspose2d,
    followed by concatenation and DoubleConv.
    """

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
        x1 = self.up(x1)

        # Pad if needed so shapes match for concatenation
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
    """1×1 convolution mapping features to output mask."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Full UNet2D model identical to the pretraining and finetuning scripts.
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
# 5. Loss Functions & Metrics (baseline: BCE + λ·Dice)
# ============================================================

bce_loss_fn = nn.BCEWithLogitsLoss()


def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    Dice loss computed from model logits.
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def combined_segmentation_loss(logits, targets, lambda_dice=0.2):
    """
    BCE loss + λ * Dice loss.
    """
    bce = bce_loss_fn(logits, targets)
    dloss = dice_loss_from_logits(logits, targets)
    return bce + lambda_dice * dloss


def compute_dice_score(logits, targets, threshold=0.5, eps=1e-6):
    """
    Dice score after thresholding sigmoid probabilities.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def compute_iou_score(logits, targets, threshold=0.5, eps=1e-6):
    """
    Intersection-over-Union (IoU) after thresholding predictions.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


# ============================================================
# 6. Build train/val/test splits from JSON (same as finetune)
# ============================================================

def build_splits_from_json(images_dir, split_json, seed=42):
    """
    Replicate finetune split logic:
      - use split_json["train"] and ["val"] lists
      - match IDs with available image files
      - shuffle val set and split 50% val / 50% test
    """
    images_dir = Path(images_dir)

    all_image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    )

    with open(split_json, "r") as f:
        split_data = json.load(f)

    def get_files_from_ids(id_list, available_files):
        file_map = {os.path.splitext(f)[0]: f for f in available_files}
        res = []
        for i in id_list:
            try:
                base = f"{int(i):04d}"   # zero-padding
            except Exception:
                base = str(i)
            if base in file_map:
                res.append(file_map[base])
        return res

    train_files = get_files_from_ids(split_data["train"], all_image_files)
    val_all_files = get_files_from_ids(split_data["val"], all_image_files)

    random.seed(seed)
    random.shuffle(val_all_files)
    split_point = len(val_all_files) // 2
    val_files = val_all_files[:split_point]
    test_files = val_all_files[split_point:]

    return train_files, val_files, test_files


# ============================================================
# 7. Load checkpoint (supports both pretrain & finetune)
# ============================================================

def load_model_weights(model, ckpt_path, device):
    """
    Generalized checkpoint loading:
      - If checkpoint contains "model_state_dict": use it (pretrain format)
      - If it contains "state_dict": use it
      - Else treat checkpoint as raw state_dict (finetune format)
    Only parameters with matching name & shape are loaded.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    loaded_dict = {}
    matched_keys = []
    skipped_keys = []

    for k, v in state_dict.items():
        if k in model_dict and hasattr(v, "shape") and v.shape == model_dict[k].shape:
            loaded_dict[k] = v
            matched_keys.append(k)
        else:
            skipped_keys.append(k)

    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(matched_keys)} keys from checkpoint. "
          f"Skipped {len(skipped_keys)} keys.")
    if skipped_keys:
        print(f"  e.g. skipped: {skipped_keys[:5]}")

    return model


# ============================================================
# 8. Per-sample evaluation (mimicking your classmate's style)
# ============================================================

def per_sample_eval(
    model,
    dataset,
    device,
    lambda_dice=0.2,
    top_k=5,
):
    """
    Reproduce your classmate's evaluation loop:
      - model.eval()
      - iterate sample by sample
      - compute loss, Dice, IoU
      - sort by loss and print best/worst samples
    """
    model.eval()

    all_sample_losses = []
    all_sample_dices = []
    all_sample_ious = []
    all_sample_metas = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Per-sample eval", unit="sample"):
            img, mask, meta = dataset[idx]

            img_t = img.unsqueeze(0).to(device)
            mask_t = mask.unsqueeze(0).to(device)

            logits = model(img_t)
            loss = combined_segmentation_loss(logits, mask_t, lambda_dice=lambda_dice)

            dice = compute_dice_score(logits, mask_t)
            iou = compute_iou_score(logits, mask_t)

            all_sample_losses.append(loss.item())
            all_sample_dices.append(dice)
            all_sample_ious.append(iou)
            all_sample_metas.append(meta)

    # Convert to numpy
    all_sample_losses = np.array(all_sample_losses)
    all_sample_dices = np.array(all_sample_dices)
    all_sample_ious = np.array(all_sample_ious)

    # Overall metrics
    mean_loss = float(all_sample_losses.mean())
    mean_dice = float(all_sample_dices.mean())
    mean_iou = float(all_sample_ious.mean())

    print("\n=== Overall metrics on this split ===")
    print(f"Mean Loss: {mean_loss:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU : {mean_iou:.4f}")

    # Sort samples
    sorted_indices = np.argsort(all_sample_losses)
    top_best = sorted_indices[:top_k]
    top_worst = sorted_indices[-top_k:][::-1]

    print(f"\n=== Top {top_k} BEST samples (lowest loss) ===")
    for rank, i in enumerate(top_best, start=1):
        meta = all_sample_metas[i]
        print(
            f"[BEST #{rank}] idx={i}, "
            f"loss={all_sample_losses[i]:.4f}, "
            f"dice={all_sample_dices[i]:.4f}, "
            f"iou={all_sample_ious[i]:.4f}, "
            f"img={meta}"
        )

    print(f"\n=== Top {top_k} WORST samples (highest loss) ===")
    for rank, i in enumerate(top_worst, start=1):
        meta = all_sample_metas[i]
        print(
            f"[WORST #{rank}] idx={i}, "
            f"loss={all_sample_losses[i]:.4f}, "
            f"dice={all_sample_dices[i]:.4f}, "
            f"iou={all_sample_ious[i]:.4f}, "
            f"img={meta}"
        )


# ============================================================
# 9. Main workflow: build dataset + load model + run eval
# ============================================================

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    # 1) Build train/val/test split using JSON (same as finetune)
    if args.split_json:
        train_files, val_files, test_files = build_splits_from_json(
            images_dir=args.images_dir,
            split_json=args.split_json,
            seed=args.seed,
        )
        if args.split == "train":
            target_files = train_files
        elif args.split == "val":
            target_files = val_files
        else:
            target_files = test_files
        print(f"Split from JSON: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    else:
        # Optional random split (not recommended for comparison to baselines)
        images_dir = Path(args.images_dir)
        all_image_files = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )
        random.seed(args.seed)
        random.shuffle(all_image_files)
        val_count = int(len(all_image_files) * args.val_ratio)
        test_count = int(len(all_image_files) * args.test_ratio)
        train_count = len(all_image_files) - val_count - test_count

        train_files = all_image_files[:train_count]
        val_files = all_image_files[train_count:train_count + val_count]
        test_files = all_image_files[train_count + val_count:]

        if args.split == "train":
            target_files = train_files
        elif args.split == "val":
            target_files = val_files
        else:
            target_files = test_files

        print(f"Random Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    print(f"Evaluating split={args.split}, num_samples={len(target_files)}")

    # 2) Build dataset with test transforms
    stage = "test" if args.split in ["val", "test"] else "train"
    transform = get_transforms(stage)
    dataset = ThyroidDataset(
        image_ids=target_files,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        transform=transform,
    )

    # 3) Build model and load weights
    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        bilinear=not args.no_bilinear,
    ).to(device)

    model = load_model_weights(model, args.checkpoint, device)

    # 4) Run per-sample evaluation
    per_sample_eval(
        model=model,
        dataset=dataset,
        device=device,
        lambda_dice=args.lambda_dice,
        top_k=args.top_k,
    )


# ============================================================
# 10. Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-sample evaluation of UNet2D segmentation on TG3K "
                    "(aligned with your finetune + classmate's evaluation style)."
    )

    # Paths
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing thyroid images")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="Directory containing thyroid masks")
    parser.add_argument("--split-json", type=str, default=None,
                        help="Path to tg3k-trainval.json (same as finetune)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (pretrained or finetuned)")

    # Which split to evaluate
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])

    # Model architecture (must match training)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--no-bilinear", action="store_true",
                        help="Use ConvTranspose instead of bilinear upsampling")

    # Loss parameter λ
    parser.add_argument("--lambda-dice", type=float, default=0.2)

    # Number of samples to display
    parser.add_argument("--top-k", type=int, default=5)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # Random split ratios (used only if split-json is missing)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

