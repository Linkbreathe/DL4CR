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
# Model: 2D U-Net (same as MIM)
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
    """Standard 2D U-Net for grayscale reconstruction."""

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
# Dataset for full-image reconstruction pretraining
# -----------------------------
def build_fullimage_pretrain_transform(height: int, width: int) -> A.Compose:
    """Augmentation pipeline for full-image masked reconstruction pretraining."""
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
    """Generate a random block-wise mask."""
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

        if current_area >= height * width:
            break

    return mask


class TG3KFullImagePretrainDataset(Dataset):
    """Full-image masked reconstruction dataset for TG3K.

    This dataset:
        1. Loads ultrasound images.
        2. Applies augmentation.
        3. Generates a random block mask and creates a masked input.
        4. Returns (masked_img, target_img, mask_tensor, meta).
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
        img_np = load_grayscale_image(img_path)
        img_hwc = np.expand_dims(img_np, axis=-1)

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
def l1_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Full-image L1 reconstruction loss."""
    return torch.mean(torch.abs(pred - target))


# -----------------------------
# Training helpers
# -----------------------------
def get_image_paths(
    images_dir: Path,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> List[Path]:
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


def load_checkpoint_for_init(
    ckpt_path: Path,
    model: nn.Module,
    map_location: Optional[str] = None,
) -> None:
    """Load only model weights from a checkpoint (for initialization)."""
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


def train_fullimage(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    images_dir = Path(args.images_dir).expanduser().resolve()
    print("Images directory:", images_dir)

    image_paths = get_image_paths(images_dir)
    print(f"Found {len(image_paths)} images for full-image pretraining.")

    transform = build_fullimage_pretrain_transform(args.img_height, args.img_width)
    dataset = TG3KFullImagePretrainDataset(
        image_paths=image_paths,
        transform=transform,
        mask_ratio=args.mask_ratio,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
    )

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

    # Optional: initialize from a previous MIM checkpoint
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

    # Optional: resume training from a full-image checkpoint
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
    global_step = start_epoch * len(loader)
    best_loss = float("inf")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (masked_img, target_img, mask_tensor, meta) in enumerate(loader):
            masked_img = masked_img.to(device)
            target_img = target_img.to(device)

            optimizer.zero_grad()
            pred = model(masked_img)
            loss = l1_reconstruction_loss(pred, target_img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            writer.add_scalar("train/loss", loss.item(), global_step)

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = running_loss / max(1, num_batches)
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Batch [{batch_idx+1}/{len(loader)}] "
                    f"Loss: {loss.item():.6f} (avg: {avg_loss:.6f})"
                )

        avg_epoch_loss = running_loss / max(1, num_batches)
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}] finished. Avg loss: {avg_epoch_loss:.6f}")

        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        latest_path = save_checkpoint(latest_state, checkpoint_dir, "fullimage_latest.pth")
        print("Saved latest checkpoint to:", latest_path)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = save_checkpoint(latest_state, checkpoint_dir, "fullimage_best.pth")
            print(f"New best model with loss {best_loss:.6f}, saved to: {best_path}")

    writer.close()
    print("Full-image pretraining finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-image masked reconstruction pretraining with 2D U-Net on TG3K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing TG3K ultrasound images (PNG/JPG).")
    parser.add_argument("--checkpoint-dir", type=str, default="./tg3k/checkpoints_fullimage",
                        help="Where to save checkpoints.")
    parser.add_argument("--log-dir", type=str, default="./tg3k/runs/fullimage_tg3k",
                        help="TensorBoard log directory.")

    parser.add_argument("--img-height", type=int, default=224,
                        help="Resize height for input images.")
    parser.add_argument("--img-width", type=int, default=320,
                        help="Resize width for input images.")

    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base number of feature maps in the U-Net.")
    parser.add_argument("--no-bilinear", action="store_true",
                        help="Use transposed conv instead of bilinear upsampling in U-Net.")

    parser.add_argument("--mask-ratio", type=float, default=0.55,
                        help="Approximate ratio of masked pixels.")
    parser.add_argument("--min-block-size", type=int, default=32,
                        help="Minimum masking block size.")
    parser.add_argument("--max-block-size", type=int, default=96,
                        help="Maximum masking block size.")

    parser.add_argument("--init-from", type=str, default=None,
                        help="Optional path to a MIM checkpoint to initialize from.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume full-image pretraining from.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="How many batches to wait before logging to stdout.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fullimage(args)
# python tg3k_MIM_Pretrain.py --images-dir ./tg3k/tg3k/thyroid-image