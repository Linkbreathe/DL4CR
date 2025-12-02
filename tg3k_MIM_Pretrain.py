import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import (
    set_seed,
    build_ultrasound_transform,
    TG3KMaskedReconstructionDataset,
    get_image_paths,
    save_checkpoint,
    load_checkpoint_for_resume,
    masked_l1_loss,
    create_unet_model,
    visualize_samples,
)


def train_mim(args: argparse.Namespace) -> None:
    # 固定随机种子
    set_seed(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    # 数据路径
    images_dir = Path(args.images_dir).expanduser().resolve()
    print("Images directory:", images_dir)

    image_paths = get_image_paths(images_dir)
    print(f"Found {len(image_paths)} images for MIM pretraining.")

    # 图像增强（统一使用 utils 中的构建函数）
    transform = build_ultrasound_transform(args.img_height, args.img_width)

    # 数据集（统一使用 TG3KMaskedReconstructionDataset）
    dataset = TG3KMaskedReconstructionDataset(
        image_paths=image_paths,
        transform=transform,
        mask_ratio=args.mask_ratio,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
    )

    # 可选：训练前可视化几张 masked 样本
    if args.visualize:
        visualize_samples(dataset, num_samples=3)

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 模型（统一通过 create_unet_model 创建）
    model = create_unet_model(
        model_type=args.model_type,
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        bilinear=not args.no_bilinear,
        device=device,
    )

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # checkpoint 目录
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 恢复训练（可选）
    start_epoch = 0
    if args.resume is not None:
        ckpt_path = Path(args.resume).expanduser().resolve()
        if ckpt_path.is_file():
            start_epoch = load_checkpoint_for_resume(
                ckpt_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                map_location=str(device),
            )
        else:
            print(f"WARNING: resume checkpoint {ckpt_path} does not exist, starting from scratch.")

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    global_step = start_epoch * len(loader)
    best_loss = float("inf")

    print("Starting MIM pretraining...")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (masked_img, target_img, mask_tensor, meta) in enumerate(loader):
            masked_img = masked_img.to(device)   # (B, 1, H, W)
            target_img = target_img.to(device)   # (B, 1, H, W)
            mask_tensor = mask_tensor.to(device) # (B, 1, H, W)

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

        # 保存 latest checkpoint
        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        latest_path = save_checkpoint(latest_state, checkpoint_dir, "mim_latest.pth")
        print("Saved latest checkpoint to:", latest_path)

        # 保存 best checkpoint
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

    parser.add_argument("--model-type", type=str, default="unet",
                        choices=["unet", "scag", "nac"],
                        help="Choose model architecture: unet, scag, or nac (default: unet)")

    parser.add_argument("--img-height", type=int, default=224,
                        help="Resize height for input images.")
    parser.add_argument("--img-width", type=int, default=320,
                        help="Resize width for input images.")

    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
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

    # Use --visualize to enable sample visualization before training
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Visualize sample masked images before training starts.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_mim(args)

# # scag
# # python tg3k_MIM_Pretrain.py --images-dir ./tg3k/tg3k/thyroid-image --checkpoint-dir ./tg3k/checkpoints_scag_mim --log-dir ./tg3k/runs/mim_tg3k_scag --model-type scag
#
# # nac
# # python tg3k_MIM_Pretrain.py --images-dir ./tg3k/tg3k/thyroid-image --checkpoint-dir ./tg3k/checkpoints_nac_mim --log-dir ./tg3k/runs/mim_tg3k_nac --model-type nac
