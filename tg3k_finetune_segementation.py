import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # Import for TensorBoard
from tqdm import tqdm


# -----------------------------
# 1. Configuration & Utils
# -----------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    Includes Python random, NumPy, PyTorch CPU/GPU, and CuDNN deterministic settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# 2. Custom Transforms (Aligned with Notebook)
# -----------------------------

class Crop_noise(A.DualTransform):
    """
    [Critical] Custom transform: Crops the top `crop_height` pixels from the image and mask.

    Reason: Ultrasound images often contain machine metadata, text, or artifacts at the top.
    The Baseline Notebook explicitly removes this (20 pixels). We must replicate this
    to ensure the input data distribution is identical.
    """

    def __init__(self, crop_height=0, always_apply=False, p=1.0):
        super(Crop_noise, self).__init__(always_apply, p)
        self.crop_height = crop_height

    def apply(self, img, **params):
        # Crop top of the image
        return img[self.crop_height:, :]

    def apply_to_mask(self, img, **params):
        # Crop top of the mask (maintain spatial alignment)
        return img[self.crop_height:, :]

    def get_transform_init_args_names(self):
        return ("crop_height",)


def get_transforms(stage: str):
    """
    Returns the Albumentations transform pipeline.

    Args:
        stage: 'train', 'valid', or 'test'
    """
    # === Baseline Core Parameters ===
    CROP_HEIGHT = 20  # Pixels to crop from top
    IMG_HEIGHT = 224  # Fixed input height
    IMG_WIDTH = 320  # Fixed input width
    MEAN = (0.5,)  # Grayscale normalization mean
    STD = (0.25,)  # Grayscale normalization std

    if stage == "train":
        return A.Compose([
            Crop_noise(crop_height=CROP_HEIGHT, p=1.0),  # Remove artifacts
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),  # Unified size
            A.HorizontalFlip(p=0.5),  # Augmentation
            A.Rotate(limit=5, p=0.5),  # Augmentation
            A.Normalize(mean=MEAN, std=STD),  # Normalization
            ToTensorV2(),  # Convert to Tensor
        ])

    elif stage in ["valid", "test"]:
        # Note: Baseline performs Crop_noise on validation/test sets too.
        # We strictly follow this to match the baseline inputs.
        return A.Compose([
            Crop_noise(crop_height=CROP_HEIGHT, p=1.0),
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    else:
        raise ValueError(f"Unknown stage: {stage}")


# -----------------------------
# 3. Dataset Implementation
# -----------------------------

class ThyroidDataset(Dataset):
    def __init__(
            self,
            image_ids: List[str],  # List of filenames (IDs)
            images_dir: str,  # Directory for images
            masks_dir: str,  # Directory for masks
            transform: Optional[A.Compose] = None,  # Augmentation pipeline
    ):
        self.image_ids = image_ids
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # 1. Construct full paths (auto-detect extensions)
        img_path = self._find_file(self.images_dir, img_id)
        mask_path = self._find_file(self.masks_dir, img_id)

        # 2. Load Image (Force 'L' mode grayscale)
        image = np.array(Image.open(img_path).convert("L"))

        # 3. Load Mask (Force 'L' mode grayscale)
        mask = np.array(Image.open(mask_path).convert("L"))

        # 4. Apply Transforms (Albumentations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 5. Mask Post-processing: Binarization
        # Even if Resize interpolation creates decimals (e.g., 0.3), we must threshold back to 0 or 1.
        if isinstance(mask, torch.Tensor):
            mask = mask.float() / 255.0 if mask.max() > 1.0 else mask.float()
            mask = (mask > 0.5).float()  # Thresholding
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # Add channel dim: (H,W) -> (1,H,W)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)
            mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return image, mask, str(img_path)

    def _find_file(self, folder: Path, filename: str) -> Path:
        """Helper: Find file with common extensions if ID doesn't have one."""
        if (folder / filename).exists():
            return folder / filename
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if (folder / (filename + ext)).exists():
                return folder / (filename + ext)
        raise FileNotFoundError(f"Could not find file for ID {filename} in {folder}")


# -----------------------------
# 4. Model Architecture (U-Net) - Standard Implementation
# -----------------------------

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Maxpool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling -> Concat -> DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle padding to ensure sizes match for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 Convolution"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # === Hardcoded Standard Channels for Baseline Alignment ===
        # Starting with 64 channels is crucial for fair model capacity comparison.
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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


# -----------------------------
# 5. Training Logic & Weight Loading
# -----------------------------

def dice_coeff(pred, target, smooth=1e-5):
    """Calculate Dice Coefficient (Evaluation Metric)"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)


def load_pretrained_weights(model, path, device):
    """
    [Critical] Robust weight loading function.

    Handles MIM pretraining weights:
    1. Automatically handles 'state_dict' or 'model' keys in checkpoint.
    2. Uses strict=False to load only matching layers (e.g., Encoder) and ignore mismatches (e.g., Decoder).
    """
    print(f"Loading weights from {path}...")
    checkpoint = torch.load(path, map_location=device)

    # 1. Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    # 2. Filter: Keep only keys with matching names AND shapes
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    # 3. Log missing keys (for debugging)
    missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())

    if len(pretrained_dict) == 0:
        print("WARNING: No matching keys found! Model is random initialized.")
    else:
        print(f"Loaded {len(pretrained_dict)} matching keys.")
        if len(missing_keys) > 0:
            print(f"Missing keys (Random Init): {len(missing_keys)} keys (e.g., {list(missing_keys)[:3]}...)")

    # 4. Update weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def train_and_evaluate(args):
    # Initialize seed
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # === Data Splitting Logic ===
    all_image_files = sorted(os.listdir(args.images_dir))
    all_image_files = [f for f in all_image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if args.split_json:
        # Mode A: Use JSON file (Aligned with Baseline)
        print(f"Loading split from {args.split_json}...")
        with open(args.split_json, 'r') as f:
            split_data = json.load(f)

        def get_files_from_ids(id_list, available_files):
            # Create map for fast lookup
            file_map = {os.path.splitext(f)[0]: f for f in available_files}
            res = []
            for i in id_list:
                # [Fix] Force convert ID to string to prevent os.path errors with int IDs
                i = str(i)
                base = os.path.splitext(i)[0]
                if base in file_map:
                    res.append(file_map[base])
            return res

        train_files = get_files_from_ids(split_data['train'], all_image_files)
        val_all_files = get_files_from_ids(split_data['val'], all_image_files)

        # [Alignment] Subsample Training set to 500
        if len(train_files) > 500:
            print(f"Subsampling training set from {len(train_files)} to 500 (random_state={args.seed}).")
            random.seed(args.seed)
            train_files = random.sample(train_files, 500)

        # [Alignment] Split JSON Val into Validation and Test (50/50 split)
        random.seed(args.seed)
        random.shuffle(val_all_files)
        split_point = len(val_all_files) // 2
        val_files = val_all_files[:split_point]
        test_files = val_all_files[split_point:]

        print(f"Data Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    else:
        # Mode B: Random Split (Fallback)
        print("No split JSON provided. Using random split.")
        random.shuffle(all_image_files)
        val_count = int(len(all_image_files) * args.val_ratio)
        test_count = int(len(all_image_files) * args.test_ratio)
        train_count = len(all_image_files) - val_count - test_count

        train_files = all_image_files[:train_count]
        val_files = all_image_files[train_count:train_count + val_count]
        test_files = all_image_files[train_count + val_count:]

    # === Create DataLoaders ===
    train_ds = ThyroidDataset(train_files, args.images_dir, args.masks_dir, transform=get_transforms("train"))
    val_ds = ThyroidDataset(val_files, args.images_dir, args.masks_dir, transform=get_transforms("valid"))
    test_ds = ThyroidDataset(test_files, args.images_dir, args.masks_dir, transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    # === Model Init ===
    model = UNet(n_channels=1, n_classes=1).to(device)

    # === Load Pretrained Weights (Key Step) ===
    if args.init_from:
        # Calls the robust loading function (allows partial loading like MIM encoder)
        model = load_pretrained_weights(model, args.init_from, device)

    # === Resume Training ===
    if args.resume:
        print(f"Resuming training from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)  # Strict load required for resume

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()  # Numerical stability with Sigmoid + BCE

    # === [Added] TensorBoard Custom Path Logic ===
    if args.tensorboard_dir:
        log_dir = Path(args.tensorboard_dir)
    else:
        log_dir = args.save_dir / "runs"

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to {log_dir}")

    best_val_dice = 0.0
    global_step = 0  # Track total steps for smooth TensorBoard curves

    # === Training Loop ===
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
            for images, masks, _ in train_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)

                # Forward
                logits = model(images)
                loss = criterion(logits, masks)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics logging
                epoch_loss += loss.item()

                # TensorBoard: Log Step Loss
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                global_step += 1

                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(1)

        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)

        # === Validation ===
        val_dice = evaluate(model, val_loader, device)

        # TensorBoard: Log Epoch Metrics
        writer.add_scalar('Train/AvgLoss', avg_train_loss, epoch)
        writer.add_scalar('Val/Dice', val_dice, epoch)

        print(f"Epoch {epoch} finished. Train Loss: {avg_train_loss:.4f}, Val Dice: {val_dice:.4f}")

        # Save Best Model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), args.save_dir / "best_model.pth")
            print(f"New best model saved! Dice: {best_val_dice:.4f}")

    # Close the writer
    writer.close()
    print("Training finished.")


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluation function: Calculates Dice Coefficient"""
    model.eval()
    dice_score = 0
    steps = 0
    for images, masks, _ in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        logits = model(images)
        probs = torch.sigmoid(logits)  # Convert to probability [0, 1]
        preds = (probs > 0.5).float()  # Threshold to binary [0, 1]

        dice_score += dice_coeff(preds, masks).item()
        steps += 1

    return dice_score / steps if steps > 0 else 0.0


# -----------------------------
# 6. Argument Parsing
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Segmentation Model (Aligned with TG3K Baseline)")

    # Path arguments
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--masks-dir", type=str, required=True, help="Directory containing masks")
    parser.add_argument("--split-json", type=str, default=None, help="Path to tg3k-trainval.json (Recommended)")
    parser.add_argument("--save-dir", type=Path, default=Path("./tg3k/checkpoints_finetune"),
                        help="Directory to save checkpoints")

    # [Added] TensorBoard Directory Argument
    parser.add_argument("--tensorboard-dir", type=str, default=None,
                        help="Specific directory for TensorBoard logs (Optional)")

    # Pretraining & Resume
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to pretraining checkpoint (e.g., MIM weights)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume fine-tuning from")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4, help="Default is 4 (Matches Baseline)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    # Fallback splitting args (Used if no JSON is provided)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)

#  python tg3k_finetune_segementation.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --init-from ./tg3k/checkpoints_mim/mim_best.pth --tensorboard-dir ./tg3k/runs/mim_finetune_tg3k --save-dir ./tg3k/
#  python tg3k_finetune_segementation.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --init-from ./tg3k/checkpoints_fullimage/fullimage_best.pth --tensorboard-dir ./tg3k/runs/full_image_finetune_tg3k --save-dir ./tg3k