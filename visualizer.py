import torch
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A

from pathlib import Path
import cv2
TARGET_HEIGHT = 224
TARGET_WIDTH = 320
from tg3k_eval import combined_segmentation_loss
from utils import load_image_as_array
def get_top5_best_and_worst_samples(model, ds):
    all_sample_losses = []
    all_sample_metas = []

    with torch.no_grad():
        for idx in range(len(ds)):
            img, mask, meta = ds[idx]

            # Add batch dimension and move to device
            img_t = img.unsqueeze(0).to(device)
            mask_t = mask.unsqueeze(0).to(device)

            logits = model(img_t)
            loss = combined_segmentation_loss(logits, mask_t, lambda_dice=0.2)

            all_sample_losses.append(loss.item())
            all_sample_metas.append(meta)
    # Get indices sorted by loss (ascending for best, descending for worst)
    # Convert list to numpy array for easy sorting
    sorted_indices = np.argsort(all_sample_losses)

    # Top 5 best performing (lowest loss)
    top_5_best_indices = sorted_indices[:5]
    # Top 5 worst performing (highest loss)
    top_5_worst_indices = sorted_indices[-5:][::-1]  # Reverse to get highest loss first

    print("Indices of 5 best performing samples (lowest loss):", top_5_best_indices)
    print("Indices of 5 worst performing samples (highest loss):", top_5_worst_indices)

    # Retrieve metadata for the best and worst samples
    best_sample_metas = [all_sample_metas[i] for i in top_5_best_indices]
    worst_sample_metas = [all_sample_metas[i] for i in top_5_worst_indices]

    print("\nMetadata for 5 best performing samples:")
    for i, meta in enumerate(best_sample_metas):
        print(f"Sample {i + 1}: Loss = {all_sample_losses[top_5_best_indices[i]]:.4f}")

    print("\nMetadata for 5 worst performing samples:")
    for i, meta in enumerate(worst_sample_metas):
        print(f"Sample {i + 1}: Loss = {all_sample_losses[top_5_worst_indices[i]]:.4f}")
    return best_sample_metas, worst_sample_metas

def visualize_samples_in_one_plot(ds, model, sample_metas, title_prefix="top 5 best or worst sample visualizations"):
    '''
    usage example:
    best_sample_metas, worst_sample_metas = get_top5_best_and_worst_samples(model, val_ds)
    visualize_samples_in_one_plot(val_ds, model, best_sample_metas, title_prefix="Top 5 Best Samples")
    '''
    resize_transform_only = A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH)

    num_samples = len(sample_metas)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    fig.suptitle(title_prefix, fontsize=18)

    if num_samples == 1:
        axes = axes.reshape(1, 4)

    for row, meta in enumerate(sample_metas):

        # Load and resize original
        img_path = Path(meta['img_path'])
        original_img_np = load_image_as_array(img_path)
        resized_img = resize_transform_only(image=original_img_np)["image"]

        # Dataset index
        original_idx = sample_metas.index(meta)

        # Get transformed tensors
        img_t, mask_t, _ = ds[original_idx]
        img_t = img_t.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            probs = torch.sigmoid(model(img_t))
            pred_mask_np = (probs > 0.5).float().squeeze().cpu().numpy()

        gt_mask_np = mask_t.squeeze().cpu().numpy()

        # Create overlay
        base_rgb = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
        red_mask = np.zeros_like(base_rgb)
        red_mask[pred_mask_np == 1] = [255, 0, 0]
        overlay = cv2.addWeighted(base_rgb, 1, red_mask, 0.5, 0)

        # Plot row
        axes[row, 0].imshow(resized_img, cmap="gray")
        axes[row, 0].set_title(f"Sample {row+1} Input")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_mask_np, cmap="gray")
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_mask_np, cmap="gray")
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Overlay")
        axes[row, 3].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def get_saliency_map(img, mask, model):
    '''
    usage example:
    img0,mask0 = val_ds[5][0],val_ds[5][1]
    get_saliency_map(img0,mask0,model)
    '''
    img = img.unsqueeze(0).to(device)
    img.requires_grad = True
    mask = mask.to(device)

    output = model(img)

    # binary segmentation → assume output [1,1,H,W]
    target = output[0].sum()
    target.backward()

    saliency = img.grad.abs().cpu()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(img.detach().cpu().squeeze(), cmap='gray')
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(saliency.detach().cpu().squeeze(), cmap='hot')
    ax[1].set_title("Saliency")
    ax[1].axis("off")

    ax[2].imshow(mask.detach().cpu().squeeze(), cmap='gray')
    ax[2].set_title("Mask")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()