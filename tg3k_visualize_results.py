import argparse
import json
import os
import random
import sys
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
from tqdm import tqdm
import matplotlib.pyplot as plt


# -----------------------------
# 1. Reuse Model & Transforms (Strictly align with Training Script)
# -----------------------------

class Crop_noise(A.DualTransform):
    """
    [Critical] Same crop logic as training to match input distribution.
    """

    def __init__(self, crop_height=0, always_apply=False, p=1.0):
        super(Crop_noise, self).__init__(always_apply, p)
        self.crop_height = crop_height

    def apply(self, img, **params):
        return img[self.crop_height:, :]

    def apply_to_mask(self, img, **params):
        return img[self.crop_height:, :]

    def get_transform_init_args_names(self):
        return ("crop_height",)


def get_test_transforms():
    """
    Returns the Validation/Test transforms used during training.
    """
    CROP_HEIGHT = 20
    IMG_HEIGHT = 224
    IMG_WIDTH = 320
    MEAN = (0.5,)
    STD = (0.25,)

    return A.Compose([
        Crop_noise(crop_height=CROP_HEIGHT, p=1.0),
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# --- Model Architecture (Copied exactly from finetune script) ---

class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
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
# 2. Dataset (Copied from finetune script)
# -----------------------------

class ThyroidDataset(Dataset):
    def __init__(self, image_ids, images_dir, masks_dir, transform=None):
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

        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Binarize Mask
        if isinstance(mask, torch.Tensor):
            mask = mask.float() / 255.0 if mask.max() > 1.0 else mask.float()
            mask = (mask > 0.5).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)
            mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return image, mask, str(img_path)

    def _find_file(self, folder: Path, filename: str) -> Path:
        if (folder / filename).exists():
            return folder / filename
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if (folder / (filename + ext)).exists():
                return folder / (filename + ext)
        raise FileNotFoundError(f"Could not find file for ID {filename} in {folder}")


# -----------------------------
# 3. Metrics Calculation
# -----------------------------

def calculate_metrics(pred, target, smooth=1e-5):
    """
    Computes Dice, IoU, Precision, Recall for a single batch.
    Pred and Target must be binary (0 or 1).
    """
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    # Dice
    dice = (2. * intersection + smooth) / (union + smooth)

    # IoU (Jaccard)
    # IoU = Intersection / (Area_Pred + Area_Target - Intersection)
    iou_union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (iou_union + smooth)

    # Precision = TP / (TP + FP) -> Intersection / Pred.sum
    precision = (intersection + smooth) / (pred.sum() + smooth)

    # Recall = TP / (TP + FN) -> Intersection / Target.sum
    recall = (intersection + smooth) / (target.sum() + smooth)

    return dice.item(), iou.item(), precision.item(), recall.item()


# -----------------------------
# 4. Visualization
# -----------------------------

def save_visualization(image_tensor, mask_tensor, pred_tensor, filename, save_dir):
    """
    Saves a side-by-side comparison: Input | GT | Pred | Overlay
    """
    # Denormalize Image: (x * std) + mean
    # Mean=0.5, Std=0.25 (Defined in transforms)
    img = image_tensor.squeeze().cpu().numpy()
    img = (img * 0.25) + 0.5
    img = np.clip(img, 0, 1)

    gt = mask_tensor.squeeze().cpu().numpy()
    pred = pred_tensor.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot 1: Original Image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Plot 2: Ground Truth
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(gt, cmap='jet', alpha=0.4)  # Overlay GT
    axes[1].set_title("Ground Truth (Blue/Red)")
    axes[1].axis('off')

    # Plot 3: Prediction
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(pred, cmap='spring', alpha=0.4)  # Overlay Pred
    axes[2].set_title("Prediction (Pink/Yellow)")
    axes[2].axis('off')

    plt.tight_layout()
    save_path = save_dir / f"{Path(filename).stem}_eval.png"
    plt.savefig(save_path)
    plt.close(fig)


# -----------------------------
# 5. Main Evaluation Logic
# -----------------------------

def evaluate_model(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Running evaluation on {device}...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Reconstruct Data Split (CRITICAL for valid evaluation) ---
    all_image_files = sorted(os.listdir(args.images_dir))
    all_image_files = [f for f in all_image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if args.split_json:
        print(f"Loading split from {args.split_json}...")
        with open(args.split_json, 'r') as f:
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

        val_all_files = get_files_from_ids(split_data['val'], all_image_files)

        # REPLICATE THE RANDOM SPLIT LOGIC FROM TRAINING SCRIPT
        random.seed(args.seed)
        random.shuffle(val_all_files)
        split_point = len(val_all_files) // 2
        # Train script used [:split_point] for validation, [split_point:] for testing
        test_files = val_all_files[split_point:]

        print(f"Total Validation IDs in JSON: {len(val_all_files)}")
        print(f"Evaluated Test Set Size (Unseen): {len(test_files)}")
    else:
        # Fallback random split
        print("Using random split fallback...")
        random.seed(args.seed)
        random.shuffle(all_image_files)
        val_count = int(len(all_image_files) * 0.1)
        test_count = int(len(all_image_files) * 0.1)
        train_count = len(all_image_files) - val_count - test_count
        test_files = all_image_files[train_count + val_count:]
        print(f"Evaluated Test Set Size: {len(test_files)}")

    # --- 2. DataLoader ---
    test_ds = ThyroidDataset(
        test_files,
        args.images_dir,
        args.masks_dir,
        transform=get_test_transforms()
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # --- 3. Load Model ---
    model = UNet(n_channels=1, n_classes=1).to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # # Handle state_dict if it's nested (standard practice)
    # if 'state_dict' in checkpoint:
    #     model.load_state_dict(checkpoint['state_dict'])
    # elif 'model' in checkpoint:  # Sometimes saved as 'model'
    #     model.load_state_dict(checkpoint['model'])
    # else:
    #     model.load_state_dict(checkpoint)
    # Handle state_dict if it's nested
    if 'model_state_dict' in checkpoint:  # <--- 新增这行，专门匹配你的文件结构
        print("Detected 'model_state_dict' key, unpacking...")
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        print("Detected 'state_dict' key, unpacking...")
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        print("Detected 'model' key, unpacking...")
        model.load_state_dict(checkpoint['model'])
    else:
        print("Loading checkpoint directly...")
        model.load_state_dict(checkpoint)

    model.eval()

    # --- 4. Inference Loop ---
    metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}

    print("Starting inference...")
    with torch.no_grad():
        for i, (image, mask, img_path) in enumerate(tqdm(test_loader)):
            image = image.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.float32)

            # Inference
            logits = model(image)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Calculate Metrics
            d, iou, p, r = calculate_metrics(preds, mask)
            metrics['dice'].append(d)
            metrics['iou'].append(iou)
            metrics['precision'].append(p)
            metrics['recall'].append(r)

            # Visualization (Save first N images and some random ones)
            if i < args.vis_count or random.random() < 0.05:
                save_visualization(image[0], mask[0], preds[0], Path(img_path[0]).name, vis_dir)

    # --- 5. Report ---
    print("\n" + "=" * 30)
    print("      EVALUATION RESULTS      ")
    print("=" * 30)
    print(f"Model: {Path(args.checkpoint).name}")
    print(f"Test Samples: {len(test_files)}")
    print("-" * 30)
    print(f"Dice Score : {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
    print(f"IoU Score  : {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
    print(f"Precision  : {np.mean(metrics['precision']):.4f}")
    print(f"Recall     : {np.mean(metrics['recall']):.4f}")
    print("=" * 30)
    print(f"Visualizations saved to: {vis_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Segmentation Model")

    # Paths
    parser.add_argument("--images-dir", type=str, required=True, help="Path to images")
    parser.add_argument("--masks-dir", type=str, required=True, help="Path to masks")
    parser.add_argument("--split-json", type=str, default=None, help="Path to tg3k-trainval.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_results"), help="Dir to save results")

    # Config
    parser.add_argument("--seed", type=int, default=42, help="Must match training seed!")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--vis-count", type=int, default=10, help="Number of mandatory visualizations")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)

# python tg3k_visualize_results.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --checkpoint ./tg3k/checkpoints_finetune/best_model.pth --output-dir ./tg3k/eval_results/mim_final_test --seed 42