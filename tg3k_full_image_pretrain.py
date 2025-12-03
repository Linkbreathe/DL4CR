import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import (
    set_seed,
    build_ultrasound_transform,
    TG3KMaskedReconstructionDataset,
    get_image_paths,
    save_checkpoint,
    load_checkpoint_for_init,
    load_checkpoint_for_resume,
    l1_reconstruction_loss,
    create_unet_model,
)

def get_image_paths_excluding_val(
    images_dir: Path,
    split_json: str,
) -> List[Path]:
    """
    Get all image paths under images_dir and exclude images whose IDs
    appear in the 'val' field of split_json. The remaining images are
    used for full-image pretraining.

    ID handling is consistent with finetune / MIM: numeric ID -> 4-digit
    zero-padded string.
    """
    # Get all image paths using the original helper
    all_image_paths = get_image_paths(images_dir)  # List[Path]
    file_map = {p.stem: p for p in all_image_paths}

    # Read JSON
    split_path = Path(split_json).expanduser().resolve()
    print(f"Loading split JSON from: {split_path}")
    with open(split_path, "r") as f:
        split_data = json.load(f)

    val_ids = split_data.get("val", [])
    excluded_stems = set()

    # Numeric ID -> 4-digit zero-padded; non-numeric -> str directly
    for i in val_ids:
        try:
            base = f"{int(i):04d}"
        except Exception:
            base = str(i)
        excluded_stems.add(base)

    # Val images that actually exist on disk
    excluded_paths = []
    for stem in excluded_stems:
        if stem in file_map:
            excluded_paths.append(file_map[stem])

    print(
        f"JSON contains {len(val_ids)} val IDs, "
        f"actually found {len(excluded_paths)} val images to exclude."
    )

    # Exclude val images from all_image_paths
    excluded_set = set(excluded_paths)
    remaining_paths = [p for p in all_image_paths if p not in excluded_set]

    print(
        f"Total images in {images_dir}: {len(all_image_paths)}\n"
        f"Excluded (val) images: {len(excluded_paths)}\n"
        f"Remaining images for full-image pretraining: {len(remaining_paths)}"
    )

    return remaining_paths


def train_fullimage(args: argparse.Namespace) -> None:
    # Fix random seed
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    # Data path
    images_dir = Path(args.images_dir).expanduser().resolve()
    print("Images directory:", images_dir)

    # If split-json is provided, exclude JSON val images; otherwise use all images
    if args.split_json is not None:
        image_paths = get_image_paths_excluding_val(images_dir, args.split_json)
    else:
        image_paths = get_image_paths(images_dir)

    print(f"Found {len(image_paths)} images for full-image pretraining.")

    # Image transforms (shared with MIM)
    transform = build_ultrasound_transform(args.img_height, args.img_width)

    # Dataset (same Dataset as MIM, using the same masking logic)
    dataset = TG3KMaskedReconstructionDataset(
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

    # Model (same construction as MIM)
    model = create_unet_model(
        model_type=args.model_type,
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        bilinear=not args.no_bilinear,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0

    # Optional: initialize from a MIM checkpoint (model_type should match)
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

    # Optional: resume from a full-image checkpoint
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
    print("Starting full-image pretraining...")
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (masked_img, target_img, mask_tensor, meta) in enumerate(loader):
            # For full-image task, we only use masked_img and target_img
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
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(loader)}] "
                    f"Loss: {loss.item():.6f} (avg: {avg_loss:.6f})"
                )

        avg_epoch_loss = running_loss / max(1, num_batches)
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        print(f"Epoch [{epoch + 1}/{args.epochs}] finished. Avg loss: {avg_epoch_loss:.6f}")

        # Latest checkpoint
        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        latest_path = save_checkpoint(latest_state, checkpoint_dir, "fullimage_latest.pth")
        print("Saved latest checkpoint to:", latest_path)

        # Best checkpoint
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

    parser.add_argument(
        "--split-json",
        type=str,
        default=None,
        help=(
            "Optional split json (e.g. tg3k-trainval.json). "
            "Images whose IDs are in 'val' will be excluded from pretraining."
        ),
    )

    parser.add_argument("--model-type", type=str, default="unet",
                        choices=["unet", "scag", "nac"],
                        help="Model architecture: unet, scag, or nac.")

    parser.add_argument("--img-height", type=int, default=224,
                        help="Resize height for input images.")
    parser.add_argument("--img-width", type=int, default=320,
                        help="Resize width for input images.")

    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
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

# # unet
# # python tg3k_full_image_Pretrain.py --images-dir ./tg3k/tg3k/thyroid-image --checkpoint-dir ./tg3k/checkpoints_unet_fullimage --log-dir ./tg3k/runs/fullimage_tg3k_unet --model-type unet --split-json ./tg3k/tg3k/tg3k-trainval.json
#
# # scag
# # python tg3k_full_image_Pretrain.py --images-dir ./tg3k/tg3k/thyroid-image --checkpoint-dir ./tg3k/checkpoints_scag_fullimage --log-dir ./tg3k/runs/fullimage_tg3k_scag --model-type scag --split-json ./tg3k/tg3k/tg3k-trainval.json
#
# # nac
# # python tg3k_full_image_Pretrain.py --images-dir ./tg3k/tg3k/thyroid-image --checkpoint-dir ./tg3k/checkpoints_nac_fullimage --log-dir ./tg3k/runs/fullimage_tg3k_nac --model-type nac --split-json ./tg3k/tg3k/tg3k-trainval.json

