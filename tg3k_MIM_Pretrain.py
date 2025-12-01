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
import matplotlib.pyplot as plt


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
# Model: 2D U-Net (grayscale in/out)
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
    """Upscaling then DoubleConv.

    If `bilinear=True`, use bilinear upsampling.
    Otherwise, use ConvTranspose2d.
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
        x1 = self.up(x1)

        # Pad if needed (in case of odd dimensions)
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
    """Standard 2D U-Net.

    Here we use:
        - in_channels = 1  (grayscale ultrasound)
        - out_channels = 1 (reconstruction of grayscale image)
    """

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
# Dataset for MIM pretraining
# -----------------------------
def build_mim_pretrain_transform(height: int, width: int) -> A.Compose:
    """Light augmentation pipeline for MIM pretraining."""
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


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load an image file as grayscale uint8 numpy array of shape (H, W)."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path).convert("L")
    return np.array(img)


def generate_block_mask(
        height: int,
        width: int,
        mask_ratio: float = 0.4,
        min_block_size: int = 32,
        max_block_size: int = 96,
) -> np.ndarray:
    """Generate a random block-wise mask.

    The mask is a float array in {0, 1} of shape (H, W), where 1 denotes a masked pixel.
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

        # Safety break in case parameters are weird
        if current_area >= height * width:
            break

    return mask


class TG3KMIMDataset(Dataset):
    """Masked-image modeling dataset for TG3K.

    This dataset:
        1. Takes a list of image paths (ultrasound frames).
        2. Applies augmentation (resize + mild augmentation + normalize).
        3. Generates a random block mask on the augmented image.
        4. Returns:
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
# Loss
# -----------------------------
def masked_l1_loss(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """L1 reconstruction loss computed only over masked pixels."""
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
# Visualization Helper
# -----------------------------
def visualize_samples(dataset: Dataset, num_samples: int = 3):
    """
    Visualize a few samples from the dataset to check masking.
    Shows: Target (Original), Masked Input, and the Mask itself.
    """
    print(f"\n--- Visualizing {num_samples} samples from dataset ---")
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        masked_img, target_img, mask_tensor, meta = dataset[idx]

        # Convert tensors to numpy arrays for plotting (C, H, W) -> (H, W)
        # Also undo normalization for visualization: (x * std) + mean
        # Assuming mean=0.5, std=0.25 based on transforms
        mean, std = 0.5, 0.25

        def denormalize(t):
            return (t.squeeze().numpy() * std + mean).clip(0, 1)

        target_np = denormalize(target_img)
        masked_np = denormalize(masked_img)
        mask_np = mask_tensor.squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(target_np, cmap="gray")
        axes[0].set_title("Original (Target)")
        axes[0].axis("off")

        axes[1].imshow(masked_np, cmap="gray")
        axes[1].set_title("Masked Input")
        axes[1].axis("off")

        axes[2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Mask (White=Masked)")
        axes[2].axis("off")

        plt.suptitle(f"Sample {i + 1}: {Path(meta['img_path']).name}")
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"sample_vis_{i}.png") # Uncomment to save instead of show if running headless
        # print(f"Saved visualization for sample {i} to sample_vis_{i}.png")


# -----------------------------
# Training/helpers
# -----------------------------
def get_image_paths(
        images_dir: Path,
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> List[Path]:
    """Collect all image paths under `images_dir` with allowed extensions."""
    image_paths: List[Path] = []
    for ext in exts:
        image_paths.extend(images_dir.rglob(f"*{ext}"))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise RuntimeError(f"No images found under {images_dir}")
    return image_paths


def save_checkpoint(
        state: Dict[str, Any],
        checkpoint_dir: Path,
        filename: str,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / filename
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint(
        ckpt_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None,
) -> int:
    """Load model (and optionally optimizer) state from checkpoint.

    Returns:
        start_epoch: the next epoch index (checkpoint_epoch + 1) if present, else 0.
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
    print(f"Loaded checkpoint from {ckpt_path}, resume from epoch {start_epoch}")
    return start_epoch


def train_mim(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    images_dir = Path(args.images_dir).expanduser().resolve()
    print("Images directory:", images_dir)

    image_paths = get_image_paths(images_dir)
    print(f"Found {len(image_paths)} images for MIM pretraining.")

    transform = build_mim_pretrain_transform(args.img_height, args.img_width)
    dataset = TG3KMIMDataset(
        image_paths=image_paths,
        transform=transform,
        mask_ratio=args.mask_ratio,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
    )

    # --- Visualization Step ---
    # Display sample images before training starts
    if args.visualize:
        visualize_samples(dataset, num_samples=3)
    # --------------------------

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

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

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.resume is not None:
        ckpt_path = Path(args.resume).expanduser().resolve()
        if ckpt_path.is_file():
            start_epoch = load_checkpoint(
                ckpt_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                map_location=str(device),
            )
        else:
            print(f"WARNING: resume checkpoint {ckpt_path} does not exist, starting from scratch.")

    writer = SummaryWriter(log_dir=args.log_dir)

    global_step = start_epoch * len(loader)
    best_loss = float("inf")

    print("Starting training...")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (masked_img, target_img, mask_tensor, meta) in enumerate(loader):
            masked_img = masked_img.to(device)  # (B, 1, H, W)
            target_img = target_img.to(device)  # (B, 1, H, W)
            mask_tensor = mask_tensor.to(device)  # (B, 1, H, W)

            optimizer.zero_grad()
            pred = model(masked_img)
            loss = masked_l1_loss(pred, target_img, mask_tensor)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            writer.add_scalar("train/loss", loss.item(), global_step)

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = running_loss / max(1, num_batches)
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(loader)}] "
                    f"Loss: {loss.item():.6f} (avg: {avg_loss:.6f})"
                )

        avg_epoch_loss = running_loss / max(1, num_batches)
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        print(f"Epoch [{epoch + 1}/{args.epochs}] finished. Avg loss: {avg_epoch_loss:.6f}")

        # Save latest checkpoint every epoch
        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        latest_path = save_checkpoint(latest_state, checkpoint_dir, "mim_latest.pth")
        print("Saved latest checkpoint to:", latest_path)

        # Save best checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = save_checkpoint(latest_state, checkpoint_dir, "mim_best.pth")
            print(f"New best model with loss {best_loss:.6f}, saved to: {best_path}")

    writer.close()
    print("MIM pretraining finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIM pretraining with 2D U-Net on TG3K ultrasound images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing TG3K ultrasound images (PNG/JPG).")
    parser.add_argument("--checkpoint-dir", type=str, default="./tg3k/checkpoints_mim",
                        help="Where to save checkpoints.")
    parser.add_argument("--log-dir", type=str, default="./tg3k/runs/mim_tg3k",
                        help="TensorBoard log directory.")

    parser.add_argument("--img-height", type=int, default=224,
                        help="Resize height for input images.")
    parser.add_argument("--img-width", type=int, default=320,
                        help="Resize width for input images.")

    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-5, help="Weight decay.")
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base number of feature maps in the U-Net.")
    parser.add_argument("--no-bilinear", action="store_true",
                        help="Use transposed conv instead of bilinear upsampling in U-Net.")

    parser.add_argument("--mask-ratio", type=float, default=0.55,
                        help="Approximate ratio of masked pixels in MIM.")
    parser.add_argument("--min-block-size", type=int, default=32,
                        help="Minimum block size (in pixels) for random masking.")
    parser.add_argument("--max-block-size", type=int, default=96,
                        help="Maximum block size (in pixels) for random masking.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (optional).")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="How many batches to wait before logging to stdout.")

    # New argument to enable visualization
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Visualize sample masked images before training starts.")

    return parser.parse_args()


if __name__ == "__main__":
    # Example usage with visualization enabled by default (or use --visualize flag)
    # To run this in a Jupyter notebook, you might need to mock args or call train_mim directly.
    # For script usage:
    args = parse_args()
    train_mim(args)

# python tg3k_MIM_Pretrain.py --images-dir ./tg3k/tg3k/thyroid-image