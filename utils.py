"""
utils.py

Common utilities for TG3K MIM / full-image pretraining and evaluation:
- Random seed setup
- Image loading & path collection
- Albumentations transforms
- Block-wise masking
- TG3K masked reconstruction dataset
- Reconstruction losses
- UNet model factory
- Checkpoint save/load helpers
- Simple visualization for sanity-checking masking
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union

import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import matplotlib.pyplot as plt

from models import UNet2D, UNet2D_scAG, UNet2D_NAC


# -----------------------------
# Global constants
# -----------------------------
DEFAULT_MEAN: float = 0.5
DEFAULT_STD: float = 0.25
DEFAULT_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# -----------------------------
# Randomness / Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Image I/O & Path helpers
# -----------------------------
def load_grayscale_image(path: Path) -> np.ndarray:
    """Load an image file as grayscale uint8 numpy array of shape (H, W)."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path).convert("L")
    return np.array(img)


def load_image_as_array(path: Union[str, Path]) -> np.ndarray:
    """Backward-compatible helper used in evaluation/visualization scripts.

    Loads a grayscale ultrasound image as uint8 array of shape (H, W),
    suitable for directly feeding into Albumentations transforms like A.Resize.

    Equivalent to load_grayscale_image, but accepts str or Path.
    """
    path = Path(path)
    return load_grayscale_image(path)




def get_image_paths(
    images_dir: Path,
    exts: Tuple[str, ...] = DEFAULT_EXTS,
) -> List[Path]:
    """Collect all image paths under `images_dir` with allowed extensions."""
    image_paths: List[Path] = []
    for ext in exts:
        image_paths.extend(images_dir.rglob(f"*{ext}"))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise RuntimeError(f"No images found under {images_dir}")
    return image_paths


# -----------------------------
# Mask generation
# -----------------------------
def generate_block_mask(
    height: int,
    width: int,
    mask_ratio: float = 0.4,
    min_block_size: int = 32,
    max_block_size: int = 96,
) -> np.ndarray:
    """Generate a random block-wise mask (1 = masked, 0 = visible).

    The expected ratio of masked pixels is roughly `mask_ratio`.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    target_area = height * width * mask_ratio
    current_area = 0.0

    while current_area < target_area:
        block_h = random.randint(min_block_size, max_block_size)
        block_w = random.randint(min_block_size, max_block_size)
        top = random.randint(0, max(0, height - block_h))
        left = random.randint(0, max(0, width - block_w))
        mask[top: top + block_h, left: left + block_w] = 1.0
        current_area = float(mask.sum())

        # Safety: avoid infinite loop if parameters are weird
        if current_area >= height * width:
            break

    return mask


# -----------------------------
# Albumentations transform
# -----------------------------
def build_ultrasound_transform(
    height: int,
    width: int,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
) -> A.Compose:
    """Common augmentation pipeline for TG3K ultrasound images."""
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
            # 可按需打开以下增强：
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.1,
            #     contrast_limit=0.1,
            #     p=0.3,
            # ),
            # A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(mean,), std=(std,)),
            ToTensorV2(),
        ]
    )


# -----------------------------
# Dataset
# -----------------------------
class TG3KMaskedReconstructionDataset(Dataset):
    """Masked reconstruction dataset for TG3K.

    统一为一个 Dataset，兼容：
        - MIM 任务（只在 mask 上算 loss）
        - Full-image 任务（全图算 loss）

    返回:
        - masked_img  : FloatTensor (1, H, W), masked input image.
        - target_img  : FloatTensor (1, H, W), full augmented image.
        - mask_tensor : FloatTensor (1, H, W), mask with 1 = masked, 0 = visible.
        - meta        : dict with metadata.
    """

    def __init__(
        self,
        image_paths: List[Path],
        transform: A.Compose,
        mask_ratio: float = 0.4,
        min_block_size: int = 32,
        max_block_size: int = 96,
    ):
        self.image_paths = list(sorted(image_paths))
        self.transform = transform
        self.mask_ratio = mask_ratio
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img_np = load_grayscale_image(img_path)  # (H, W), uint8

        # Albumentations expects HWC images
        img_hwc = np.expand_dims(img_np, axis=-1)  # (H, W, 1)

        augmented = self.transform(image=img_hwc)
        img_tensor = augmented["image"].float()  # (1, H, W)

        _, h, w = img_tensor.shape
        mask_np = generate_block_mask(
            height=h,
            width=w,
            mask_ratio=self.mask_ratio,
            min_block_size=self.min_block_size,
            max_block_size=self.max_block_size,
        )
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)

        target_img = img_tensor
        masked_img = target_img * (1.0 - mask_tensor)

        meta = {
            "img_path": str(img_path),
            "height": int(h),
            "width": int(w),
        }

        return masked_img, target_img, mask_tensor, meta


# -----------------------------
# Losses
# -----------------------------
def l1_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Full-image L1 reconstruction loss."""
    return torch.mean(torch.abs(pred - target))


def masked_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """L1 reconstruction loss computed only over masked pixels.

    Args:
        pred:   (B, C, H, W)
        target: (B, C, H, W)
        mask:   (B, C, H, W) with values in {0, 1}
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape}, target {target.shape}, mask {mask.shape}"
        )

    abs_diff = torch.abs(pred - target)
    masked_diff = abs_diff * mask
    denom = mask.sum()
    if denom.item() < eps:
        # Fallback: use full-image loss if mask is degenerate
        return abs_diff.mean()
    return masked_diff.sum() / (denom + eps)


# -----------------------------
# Model factory
# -----------------------------
def create_unet_model(
    model_type: str,
    in_channels: int,
    out_channels: int,
    base_channels: int,
    bilinear: bool,
    device: torch.device,
) -> nn.Module:
    """Create a UNet2D / UNet2D_scAG / UNet2D_NAC model based on model_type."""
    if model_type == "nac":
        print("Using UNet2D_NAC (NAC + scAG) ...")
        ModelClass = UNet2D_NAC
    elif model_type == "scag":
        print("Using UNet2D_scAG (scAG) ...")
        ModelClass = UNet2D_scAG
    else:
        print("Using standard UNet2D ...")
        ModelClass = UNet2D

    model = ModelClass(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        bilinear=bilinear,
    ).to(device)

    return model


# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Path,
    filename: str,
) -> Path:
    """Save checkpoint dict to `checkpoint_dir/filename`."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / filename
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint_for_init(
    ckpt_path: Path,
    model: nn.Module,
    map_location: Optional[str] = None,
) -> None:
    """Load only model weights from a checkpoint (for initialization).

    Uses strict=False, so it's tolerant to missing / unexpected keys.
    Useful when:
        - Initializing full-image pretraining from a MIM checkpoint.
        - Loading from slightly different architectures.
    """
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
    """Load model + optimizer state to resume training.

    Returns:
        start_epoch: epoch index to continue from (checkpoint_epoch + 1).
    """
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


