import argparse
from pathlib import Path
import random
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2


# -----------------------------
# Utility helpers
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model: 2D U-Net (shared with pretraining)
# -----------------------------
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


# -----------------------------
# Dataset for TG3K segmentation
# -----------------------------
def load_grayscale_image(path: Path) -> np.ndarray:
    """Load an image file as grayscale uint8 (H, W)."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path).convert("L")
    return np.array(img)


def load_mask_image(path: Path) -> np.ndarray:
    """Load a segmentation mask as uint8 (H, W). Non-zero = foreground."""
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")
    mask = Image.open(path).convert("L")
    return np.array(mask)


def build_seg_train_transform(height: int, width: int) -> A.Compose:
    """Augmentation pipeline for segmentation training."""
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.1,
            #     contrast_limit=0.1,
            #     p=0.3,
            # ),
            # A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.5,), std=(0.25,)),
            ToTensorV2(),
        ]
    )


def build_seg_val_transform(height: int, width: int) -> A.Compose:
    """Validation / test transform (no strong augmentation)."""
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.5,), std=(0.25,)),
            ToTensorV2(),
        ]
    )


class TG3KSegmentationDataset(Dataset):
    """Segmentation dataset for TG3K.

    Assumptions:
        - `images_dir` contains ultrasound images (.png/.jpg/...).
        - `masks_dir` contains binary masks with the same basename as images.

    If `ids` is None, we match image/mask pairs by basename across all images.
    Otherwise we only keep pairs whose basename is in `ids`.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        transform: A.Compose,
        ids: Optional[List[str]] = None,
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        all_images: List[Path] = []
        for ext in exts:
            all_images.extend(images_dir.rglob(f"*{ext}"))
        all_images = sorted(set(all_images))

        pairs: List[Tuple[Path, Path]] = []
        for img_path in all_images:
            base = img_path.stem
            if ids is not None and base not in ids:
                continue

            mask_path: Optional[Path] = None
            for ext in exts:
                candidate = masks_dir / f"{base}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:
                # If there is no corresponding mask, skip this sample.
                continue

            pairs.append((img_path, mask_path))

        if not pairs:
            raise RuntimeError(f"No image-mask pairs found under {images_dir} and {masks_dir}")

        self.pairs = pairs
        self.ids = [p[0].stem for p in pairs]
        print(f"TG3KSegmentationDataset: found {len(self.pairs)} pairs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img_np = load_grayscale_image(img_path)
        mask_np = load_mask_image(mask_path)

        # Binarize mask: foreground = 1, background = 0
        mask_bin = (mask_np > 0).astype(np.float32)

        # Albumentations expects HWC
        img_hwc = np.expand_dims(img_np, axis=-1)
        mask_hwc = np.expand_dims(mask_bin, axis=-1)

        augmented = self.transform(image=img_hwc, mask=mask_hwc)
        img_tensor = augmented["image"].float()          # (1, H, W)
        mask_tensor = augmented["mask"].float()
        # Ensure shape is (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[0:1, ...]
        mask_tensor = mask_tensor.clamp(0.0, 1.0)

        meta = {
            "img_path": str(img_path),
            "mask_path": str(mask_path),
            "id": img_path.stem,
        }

        return img_tensor, mask_tensor, meta


# -----------------------------
# Loss and metrics
# -----------------------------
bce_loss_fn = nn.BCEWithLogitsLoss()


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss computed from logits and target masks in {0,1}.

    Args:
        logits:  (B, 1, H, W)
        targets: (B, 1, H, W), floats in {0,1}
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    dice_loss = 1.0 - dice.mean()
    return dice_loss


def combined_segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss + Dice loss."""
    loss_bce = bce_loss_fn(logits, targets)
    loss_dice = dice_loss_from_logits(logits, targets)
    return loss_bce + loss_dice


def compute_dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """Dice coefficient on thresholded predictions."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def compute_iou_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """IoU (Jaccard) score on thresholded predictions."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Path,
    filename: str,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / filename
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint_for_init(
    ckpt_path: Path,
    model: nn.Module,
    map_location: Optional[str] = None,
) -> None:
    """Load only model weights from a checkpoint for initialization."""
    if map_location is None:
        map_location = "cpu"
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded init weights from {ckpt_path}")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)


def load_checkpoint_for_resume(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> int:
    """Load model + optimizer to resume training."""
    if map_location is None:
        map_location = "cpu"

    checkpoint = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: could not load optimizer state from checkpoint: {e}")

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    start_epoch = max(start_epoch, 0)
    print(f"Resuming from checkpoint {ckpt_path}, epoch {start_epoch}")
    return start_epoch


# -----------------------------
# Train / val / test splits
# -----------------------------
def split_ids(ids: List[str], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    """Randomly split ids into train/val/test."""
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_ids = ids[:n_test]
    val_ids = ids[n_test: n_test + n_val]
    train_ids = ids[n_test + n_val:]

    return train_ids, val_ids, test_ids


def create_datasets_and_loaders(args: argparse.Namespace, device: torch.device):
    images_dir = Path(args.images_dir).expanduser().resolve()
    masks_dir = Path(args.masks_dir).expanduser().resolve()
    print("Images directory:", images_dir)
    print("Masks directory :", masks_dir)

    # Build a dummy dataset to discover all ids
    dummy_transform = build_seg_val_transform(args.img_height, args.img_width)
    dummy_dataset = TG3KSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=dummy_transform,
        ids=None,
    )
    all_ids = dummy_dataset.ids
    print(f"Total paired samples: {len(all_ids)}")

    if args.train_ids_file or args.val_ids_file or args.test_ids_file:
        def read_ids(path: Optional[str]) -> Optional[List[str]]:
            if path is None:
                return None
            p = Path(path).expanduser().resolve()
            with open(p, "r") as f:
                return [line.strip() for line in f if line.strip()]

        train_ids = read_ids(args.train_ids_file) or all_ids
        val_ids = read_ids(args.val_ids_file) or []
        test_ids = read_ids(args.test_ids_file) or []
    else:
        train_ids, val_ids, test_ids = split_ids(
            all_ids,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    print(f"Train/Val/Test sizes: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")

    train_transform = build_seg_train_transform(args.img_height, args.img_width)
    val_transform = build_seg_val_transform(args.img_height, args.img_width)
    test_transform = build_seg_val_transform(args.img_height, args.img_width)

    train_dataset = TG3KSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=train_transform,
        ids=train_ids,
    )
    val_dataset = TG3KSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=val_transform,
        ids=val_ids if val_ids else None,  # If empty, this will reuse all ids
    )
    test_dataset = TG3KSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=test_transform,
        ids=test_ids if test_ids else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# Train / evaluate
# -----------------------------
def train_and_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, device)

    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        bilinear=not args.no_bilinear,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Optional initialization from pretraining (MIM or full-image)
    if args.init_from is not None:
        init_path = Path(args.init_from).expanduser().resolve()
        if init_path.is_file():
            load_checkpoint_for_init(
                ckpt_path=init_path,
                model=model,
                map_location=str(device),
            )
        else:
            print(f"WARNING: init_from checkpoint {init_path} does not exist, skipping initialization.")

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser().resolve()
        if resume_path.is_file():
            start_epoch = load_checkpoint_for_resume(
                ckpt_path=resume_path,
                model=model,
                optimizer=optimizer,
                map_location=str(device),
            )
        else:
            print(f"WARNING: resume checkpoint {resume_path} does not exist, starting from scratch.")

    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = start_epoch * len(train_loader)
    best_val_dice = 0.0

    for epoch in range(start_epoch, args.epochs):
        # ---- Training ----
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_idx, (imgs, masks, meta) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_segmentation_loss(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1
            global_step += 1

            writer.add_scalar("train/loss", loss.item(), global_step)

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = train_loss_sum / max(1, train_batches)
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Train loss: {loss.item():.6f} (avg: {avg_loss:.6f})"
                )

        avg_train_loss = train_loss_sum / max(1, train_batches)
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}] training finished. Avg loss: {avg_train_loss:.6f}")

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for imgs, masks, meta in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = combined_segmentation_loss(logits, masks)

                val_loss_sum += loss.item()
                val_dice_sum += compute_dice_score(logits, masks)
                val_iou_sum += compute_iou_score(logits, masks)
                val_batches += 1

        avg_val_loss = val_loss_sum / max(1, val_batches)
        avg_val_dice = val_dice_sum / max(1, val_batches)
        avg_val_iou = val_iou_sum / max(1, val_batches)

        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("val/dice", avg_val_dice, epoch)
        writer.add_scalar("val/iou", avg_val_iou, epoch)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] validation: "
            f"loss={avg_val_loss:.4f}, dice={avg_val_dice:.4f}, iou={avg_val_iou:.4f}"
        )

        # ---- Checkpointing ----
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_dice": avg_val_dice,
            "val_iou": avg_val_iou,
        }
        latest_path = save_checkpoint(state, checkpoint_dir, "finetune_latest.pth")
        print("Saved latest checkpoint to:", latest_path)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_path = save_checkpoint(state, checkpoint_dir, "finetune_best.pth")
            print(f"New best model with Dice={best_val_dice:.4f}, saved to: {best_path}")

    writer.close()

    # ---- Final test evaluation (using best model if available) ----
    best_ckpt_path = checkpoint_dir / "finetune_best.pth"
    if best_ckpt_path.is_file():
        print("Loading best checkpoint for final test evaluation:", best_ckpt_path)
        checkpoint = torch.load(best_ckpt_path, map_location=str(device))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Best checkpoint not found, using latest model for test.")

    model.eval()
    test_loss_sum = 0.0
    test_dice_sum = 0.0
    test_iou_sum = 0.0
    test_batches = 0

    with torch.no_grad():
        for imgs, masks, meta in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = combined_segmentation_loss(logits, masks)

            test_loss_sum += loss.item()
            test_dice_sum += compute_dice_score(logits, masks)
            test_iou_sum += compute_iou_score(logits, masks)
            test_batches += 1

    avg_test_loss = test_loss_sum / max(1, test_batches)
    avg_test_dice = test_dice_sum / max(1, test_batches)
    avg_test_iou = test_iou_sum / max(1, test_batches)

    print("\n=== Final test set evaluation ===")
    print(f"Test loss: {avg_test_loss:.4f}")
    print(f"Test Dice: {avg_test_dice:.4f}")
    print(f"Test IoU : {avg_test_iou:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune 2D U-Net for TG3K thyroid segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing TG3K ultrasound images.")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="Directory containing TG3K segmentation masks.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_finetune",
                        help="Where to save checkpoints.")
    parser.add_argument("--log-dir", type=str, default="./runs/finetune_tg3k",
                        help="TensorBoard log directory.")

    parser.add_argument("--img-height", type=int, default=320,
                        help="Resize height for input images.")
    parser.add_argument("--img-width", type=int, default=320,
                        help="Resize width for input images.")

    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base number of feature maps in the U-Net.")
    parser.add_argument("--no-bilinear", action="store_true",
                        help="Use transposed conv instead of bilinear upsampling in U-Net.")

    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Val ratio if no id files are provided.")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test ratio if no id files are provided.")

    parser.add_argument("--train-ids-file", type=str, default=None,
                        help="Optional text file listing image basenames for training (one per line).")
    parser.add_argument("--val-ids-file", type=str, default=None,
                        help="Optional text file listing image basenames for validation.")
    parser.add_argument("--test-ids-file", type=str, default=None,
                        help="Optional text file listing image basenames for test.")

    parser.add_argument("--init-from", type=str, default=None,
                        help="Optional path to a pretraining checkpoint (MIM or full-image).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume fine-tuning from.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="How many batches to wait before logging to stdout.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
