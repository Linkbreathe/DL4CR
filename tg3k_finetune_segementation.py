import argparse
import json
import os
import random
from pathlib import Path
from typing import List

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

# Import model definitions from your own file
from models import UNet2D, UNet2D_scAG, UNet2D_NAC


# ============================================================
# 1. Utility: random seed
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds to make experiments reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Three-stage transforms (common / image / mask)
# ============================================================

TARGET_HEIGHT = 224
TARGET_WIDTH = 320
MEAN = (0.5,)   # single-channel grayscale
STD = (0.25,)


def get_segmentation_train_transforms_split():
    """
    Three-stage transforms for the training phase.

    - common_transforms: geometric transforms applied to both image & mask
        * Resize
        * HorizontalFlip
        * Rotate
    - image_transforms: pixel-level transforms applied only to the image
        * Normalize(mean=0.5, std=0.25)
        * ToTensorV2
    - mask_transforms: transforms applied only to the mask
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
    Three-stage transforms for validation / test phase.

    - common_transforms: deterministic Resize only (no flip / rotate).
    - image_transforms: Normalize + ToTensor.
    - mask_transforms: ToFloat + ToTensor.
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
# 3. Dataset: three-stage ThyroidDataset
# ============================================================

class ThyroidDataset(Dataset):
    """
    Dataset implementation with three-stage transforms.

    Args:
        image_ids: list of file names (with or without extension).
        images_dir: directory of images.
        masks_dir: directory of segmentation masks.
        common_transforms: transforms applied jointly to image & mask.
        image_transforms: transforms applied only to the image.
        mask_transforms: transforms applied only to the mask.

    Returns:
        image: Tensor of shape (1, H, W), normalized.
        mask:  Tensor of shape (1, H, W), binarized to {0,1}.
        meta:  string path to the original image file.
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

        # Load as grayscale
        img_np = np.array(Image.open(img_path).convert("L"))   # (H, W)
        img_np = np.expand_dims(img_np, axis=-1)               # (H, W, 1) for albumentations
        mask_np = np.array(Image.open(mask_path).convert("L")) # (H, W)

        # 1) Common geometric transforms (keep image & mask aligned)
        augmented_common = self.common_transforms(image=img_np, mask=mask_np)
        img_common = augmented_common["image"]   # (H, W, 1) or (H, W)
        mask_common = augmented_common["mask"]   # (H, W)

        # 2) Image-only transforms: Normalize + ToTensorV2
        img_t = self.image_transforms(image=img_common)["image"]  # Tensor (1, H, W)

        # 3) Mask-only transforms: ToFloat + ToTensorV2
        mask_t = self.mask_transforms(image=mask_common)["image"]  # Tensor (H, W) or (1, H, W)

        # Ensure mask has a channel dimension
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)  # (H, W) -> (1, H, W)

        # Binarize mask to {0, 1}
        mask_t = (mask_t > 0.5).float()

        meta = str(img_path)
        return img_t, mask_t, meta

    def _find_file(self, folder: Path, filename: str) -> Path:
        """
        Find real file on disk for a given ID (with or without extension).
        Tries several common image extensions.
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
# 4. Loss functions & metrics (BCE + λ · Dice)
# ============================================================

def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Dice loss computed from raw logits.

    Args:
        logits: raw outputs of the model (B, 1, H, W).
        targets: ground truth masks (B, 1, H, W).
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
    Combined segmentation loss:
        L = BCEWithLogits + λ * DiceLoss
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
    """
    Compute mean Dice score over a batch using thresholded predictions.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


# ============================================================
# 5. Load weights from MIM-pretrained checkpoint
# ============================================================

def load_pretrained_weights(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """
    Load weights from a MIM-pretrained checkpoint.

    Supported formats:
      - {'model_state_dict': ...}
      - {'state_dict': ...}
      - {'model': ...}
      - plain state_dict

    Only layers whose name AND shape match are loaded.
    The remaining layers stay randomly initialized.
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
            print(
                f"{len(missing_keys)} layers remain randomly initialized "
                f"(e.g. {list(missing_keys)[:5]}...)"
            )

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


# ============================================================
# 6. Model factory: model_type consistent with pretraining
# ============================================================

def create_unet_model(
    model_type: str,
    in_channels: int,
    out_channels: int,
    base_channels: int,
    bilinear: bool,
    device: torch.device,
) -> nn.Module:
    """
    Factory to create U-Net variants consistent with MIM pretraining.

    Args:
        model_type: one of ['unet', 'scag', 'nac'].
    """
    if model_type == "unet":
        model = UNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bilinear=bilinear,
        )
    elif model_type == "scag":
        model = UNet2D_scAG(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bilinear=bilinear,
        )
    elif model_type == "nac":
        model = UNet2D_NAC(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bilinear=bilinear,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    return model


# ============================================================
# 7. Evaluation function (val/test)
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate mean Dice score over an entire DataLoader.
    """
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
# 8. Training + Early Stopping + final Test evaluation
# ============================================================

def train_and_evaluate(args):
    """
    Main training loop with early stopping and final test evaluation.
    """
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # -----------------------------
    # Collect image file names
    # -----------------------------
    all_image_files = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    )

    # -----------------------------
    # Data split: prefer split-json if provided
    # -----------------------------
    if args.split_json:
        print(f"Loading split from {args.split_json}...")
        with open(args.split_json, 'r') as f:
            split_data = json.load(f)

        def get_files_from_ids(id_list, available_files):
            """
            Map ID list (e.g. [1, 2, 3]) to actual file names based on stem.
            If an ID is numeric, it is converted to a 4-digit zero-padded string
            (e.g. 1 -> '0001').
            """
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

        # Optional: subsample train set to a fixed size
        if args.train_subset_size > 0 and len(train_files) > args.train_subset_size:
            print(
                f"Subsampling training set from {len(train_files)} to "
                f"{args.train_subset_size} (seed={args.seed})."
            )
            random.seed(args.seed)
            train_files = random.sample(train_files, args.train_subset_size)

        random.seed(args.seed)
        random.shuffle(val_all_files)
        split_point = len(val_all_files) // 2
        val_files = val_all_files[:split_point]
        test_files = val_all_files[split_point:]

        print(
            f"Data split from JSON: Train={len(train_files)}, "
            f"Val={len(val_files)}, Test={len(test_files)}"
        )
    else:
        # Fallback: random split without JSON
        print("No split JSON provided. Using random split.")
        random.seed(args.seed)
        random.shuffle(all_image_files)
        val_count = int(len(all_image_files) * args.val_ratio)
        test_count = int(len(all_image_files) * args.test_ratio)
        train_count = len(all_image_files) - val_count - test_count

        train_files = all_image_files[:train_count]
        val_files = all_image_files[train_count:train_count + val_count]
        test_files = all_image_files[train_count + val_count:]

        print(
            f"Random Split: Train={len(train_files)}, "
            f"Val={len(val_files)}, Test={len(test_files)}"
        )

    # -----------------------------
    # Build three-stage Dataset & DataLoader
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
        common_transforms=val_common,   # test uses same transforms as val (no random aug)
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
    # Initialize model (match MIM pretraining)
    # -----------------------------
    model = create_unet_model(
        model_type=args.model_type,
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        bilinear=not args.no_bilinear,
        device=device,
    )
    print(
        f"Model initialized: type={args.model_type}, "
        f"base_channels={args.base_channels}, "
        f"bilinear={not args.no_bilinear}"
    )

    # Load MIM-pretrained weights if provided
    if args.init_from:
        model = load_pretrained_weights(model, args.init_from, device)

    # Resume from full model checkpoint if provided
    if args.resume:
        print(f"Resuming training from full checkpoint: {args.resume}...")
        resume_ckpt = torch.load(args.resume, map_location=device)
        if isinstance(resume_ckpt, dict) and "state_dict" in resume_ckpt:
            model.load_state_dict(resume_ckpt["state_dict"])
        else:
            model.load_state_dict(resume_ckpt)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # -----------------------------
    # TensorBoard setup
    # -----------------------------
    if args.tensorboard_dir:
        log_dir = Path(args.tensorboard_dir)
    else:
        log_dir = Path(args.save_dir) / "runs"

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to {log_dir}")

    # -----------------------------
    # Training loop + early stopping
    # -----------------------------
    best_val_dice = 0.0
    epochs_without_improvement = 0
    global_step = 0

    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        # ---------- Train ----------
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, masks, _) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            logits = model(images)
            loss = combined_segmentation_loss(
                logits, masks, lambda_dice=args.lambda_dice
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            writer.add_scalar('train/loss_step', loss.item(), global_step)

            # Text logging similar to pretraining script
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / max(1, num_batches)
                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f} (avg: {avg_loss:.6f})"
                )

        avg_train_loss = epoch_loss / max(1, num_batches)
        writer.add_scalar('train/loss_epoch', avg_train_loss, epoch)

        # ---------- Validate (Dice) ----------
        val_dice = evaluate(model, val_loader, device)
        writer.add_scalar('val/dice', val_dice, epoch)

        print(
            f"Epoch {epoch}/{args.epochs} finished. "
            f"Train Loss: {avg_train_loss:.4f}, Val Dice: {val_dice:.4f}"
        )

        # ---------- Save best model & handle early stopping ----------
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(
                f"  -> New best model saved to {best_path} "
                f"(Val Dice: {best_val_dice:.4f})"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  -> No improvement in Dice for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    writer.close()
    print("Training finished.")

    # -----------------------------
    # Final evaluation on test set
    # -----------------------------
    if best_path.exists():
        print(f"Loading best model from {best_path} for final test evaluation...")
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print(
            "Best model file not found, using last epoch model "
            "for test evaluation."
        )

    test_dice = evaluate(model, test_loader, device)
    print(f"Final Test Dice: {test_dice:.4f}")


# ============================================================
# 9. Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune UNet2D / UNet2D_scAG / UNet2D_NAC on TG3K "
            "with MIM-pretrained weights and 3-stage transforms."
        )
    )

    # Paths
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing input images."
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=True,
        help="Directory containing segmentation masks."
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default=None,
        help="Path to tg3k-trainval.json (recommended)."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./tg3k/checkpoints_finetune",
        help="Directory to save best checkpoints."
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs (optional)."
    )

    # Pretraining & resume
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to pretraining checkpoint (e.g., MIM mim_best.pth)."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to full model checkpoint to resume from."
    )

    # Model hyperparameters (must match MIM pretraining)
    parser.add_argument(
        "--model-type",
        type=str,
        default="unet",
        choices=["unet", "scag", "nac"],
        help="Backbone type, must match MIM pretrain: unet / scag / nac."
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base number of feature maps in UNet2D (match MIM pretrain)."
    )
    parser.add_argument(
        "--no-bilinear",
        action="store_true",
        help="Use ConvTranspose instead of bilinear upsampling."
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default 4 matches baseline)."
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for Adam, e.g. 1e-5."
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="How many batches to wait before logging to stdout."
    )

    # λ in BCE + Dice
    parser.add_argument(
        "--lambda-dice",
        type=float,
        default=0.2,
        help="Weight for Dice loss term in combined loss (BCE + λ * Dice)."
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience based on validation Dice."
    )

    # Random split ratios when split_json is not provided
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    # Optional training subset size when using split_json
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=0,
        help="If >0 and using split-json, subsample this many training images."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)



# MIM
# # unet
# python tg3k_finetune_segementation.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --init-from ./tg3k/checkpoints_unet_mim/mim_best.pth --tensorboard-dir ./tg3k/runs/mim_unet_finetune_tg3k --save-dir ./tg3k/checkpoints_finetune/mim_unet --train-subset-size 500 --model-type unet
#
# # scag
# python tg3k_finetune_segementation.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --init-from ./tg3k/checkpoints_scag_mim/mim_best.pth --tensorboard-dir ./tg3k/runs/mim_scag_finetune_tg3k --save-dir ./tg3k/checkpoints_finetune/mim_scag --train-subset-size 500 --model-type scag
#
# # nac
# python tg3k_finetune_segementation.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --init-from ./tg3k/checkpoints_nac_mim/mim_best.pth --tensorboard-dir ./tg3k/runs/mim_nac_finetune_tg3k --save-dir ./tg3k/checkpoints_finetune/mim_nac --train-subset-size 500 --model-type nac


# Full Image
# # unet
# python tg3k_finetune_segementation.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --init-from ./tg3k/checkpoints_unet_fullimage/fullimage_best.pth --tensorboard-dir ./tg3k/runs/fullimage_unet_finetune_tg3k --save-dir ./tg3k/checkpoints_finetune/fullimage_unet --train-subset-size 500 --model-type unet
#
# # scag
# python tg3k_finetune_segementation.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --init-from ./tg3k/checkpoints_scag_fullimage/fullimage_best.pth --tensorboard-dir ./tg3k/runs/fullimage_scag_finetune_tg3k --save-dir ./tg3k/checkpoints_finetune/fullimage_scag --train-subset-size 500 --model-type scag
#
# # nac
# python tg3k_finetune_segementation.py --images-dir ./tg3k/tg3k/thyroid-image --masks-dir ./tg3k/tg3k/thyroid-mask --split-json ./tg3k/tg3k/tg3k-trainval.json --init-from ./tg3k/checkpoints_nac_fullimage/fullimage_best.pth --tensorboard-dir ./tg3k/runs/fullimage_nac_finetune_tg3k --save-dir ./tg3k/checkpoints_finetune/fullimage_nac --train-subset-size 500 --model-type nac




