````markdown
# ToothSeg-MIM: Self-Supervised Pretraining for Tooth Segmentation

This repository implements a **self-supervised pretraining pipeline** for 2D dental X-ray images using **Masked Image Modeling (MIM)** with a **2D U-Net backbone**, based on the **STS-2D-Tooth** dataset.

The high-level idea:

1. **Preprocess** the STS-2D-Tooth dataset and organize all 2D images.
2. Build an index of images and masks, and define subsets for:
   - **Pretraining** (all non-mask images, labeled + unlabeled)
   - (Later) **Segmentation fine-tuning** (image–mask pairs)
3. **Pretrain** a 2D U-Net to reconstruct masked-out regions (MIM) on all 2D images.
4. Save the pretrained U-Net encoder–decoder weights for downstream **tooth segmentation**.

---

## Repository Structure

Key files (as used in this project):

- `PreProcessing.ipynb`  
  Download, verify, unpack and index the STS-Tooth dataset.

- `Data_enhance.ipynb`  
  Build segmentation / pretraining subsets, basic statistics, and augmentation previews.

- `MIM_Pretrain.ipynb`  
  Self-supervised MIM pretraining using a 2D U-Net backbone.

Data & outputs:

- `sts_tooth_data/`
  - `downloads/` – raw multi-part zip files (`SD-Tooth.zip.001` … `.015`)
  - `raw/`       – extracted content from the merged `SD-Tooth.zip`
  - `processed_2d/`
    - `adult/labeled/images/`, `adult/labeled/masks/`, …
    - `children/...`
  - `sts2d_index.csv` – global index of all PNG images (2D images + masks)
  - `checkpoints/`
    - `unet2d_mim_pretrained.pth` – pretrained 2D U-Net weights
  - `runs/`
    - `mim_unet/` – TensorBoard logs for MIM pretraining

---

## 1. Dataset: STS-2D-Tooth

The code assumes you are using the **2D subset** of the STS-Tooth dataset (often referred to as **STS-2D-Tooth**).  
Data is distributed as **multi-part zip files** on Zenodo (record ID `10597292`):

- `SD-Tooth.zip.001`  
- `SD-Tooth.zip.002`  
- …  
- `SD-Tooth.zip.015`

### 1.1 Download & Extract (via `PreProcessing.ipynb`)

`PreProcessing.ipynb` will:

1. Create a data root (by default):

   ```python
   DATA_ROOT = Path("./sts_tooth_data").resolve()
````

2. Download all 15 parts into `sts_tooth_data/downloads/`.
3. Optionally verify MD5 checksums for each part.
4. Merge the 15 parts into a single `SD-Tooth.zip`.
5. Extract the merged zip into `sts_tooth_data/raw/`.

### 1.2 Build 2D Index & Processed Layout

The notebook scans the extracted folder for the **STS-2D-Tooth** directory and indexes all `.png` files, inferring:

* `rel_path` – relative path under STS-2D-Tooth
* `age_group` – `"adult"` or `"children"`
* `label_status` – `"labeled"` or `"unlabeled"`
* `is_mask` – `True` if the file is a mask, `False` otherwise
* `pair_id` – a normalized ID to match image and mask pairs

It then creates a **processed 2D layout**:

```text
sts_tooth_data/processed_2d/
  adult/
    labeled/
      images/
      masks/
    unlabeled/
      images/
  children/
    ...
```

and writes the global index:

```text
sts_tooth_data/sts2d_index.csv
```

---

## 2. Data Index & Subsets (`Data_enhance.ipynb`)

`Data_enhance.ipynb` consumes `sts2d_index.csv` and performs:

* **Global statistics**:

  ```text
  Full index shape: (N_total, 5)

  age_group:
    adult       4350
    children     550

  label_status:
    unlabeled   3100
    labeled     1800

  is_mask:
    False    4000   # images
    True      900   # masks
  ```

* **Build segmentation pairs** (image + mask) where both exist (≈ 900 pairs).

* **Build pretraining subset**:

  ```python
  df_pretrain = df[df["is_mask"] == False]  # all 4000 images
  ```

For **MIM pretraining**, we use **all 4000 non-mask images** (both labeled and unlabeled); masks are not used in the MIM objective.

---

## 3. MIM Pretraining with 2D U-Net (`MIM_Pretrain.ipynb`)

This notebook implements **Masked Image Modeling** using a **2D U-Net**:

### 3.1 Data Pipeline

1. Load `sts2d_index.csv` and build:

   ```python
   df_pretrain = df[df["is_mask"] == False].reset_index(drop=True)
   ```

2. **Transform** (Albumentations):

   * `Resize` to `(H=320, W=640)`
   * `HorizontalFlip`
   * Small `Rotate` (±5 degrees)
   * Mild `RandomBrightnessContrast`
   * Mild `GaussNoise` / `GaussianBlur`
   * `ToFloat` + `Normalize(mean=0.5, std=0.25)`
   * `ToTensorV2` → shape `(1, H, W)` (grayscale)

3. **Random block masking in pixel space**:

   For each sample, a binary mask `mask_tensor` of shape `(1, H, W)` is generated:

   ```python
   mask_np = random_block_mask(
       height=H,
       width=W,
       num_blocks=8,
       min_block_fraction=0.1,
       max_block_fraction=0.3,
   )
   ```

   * `mask == 1.0` → **masked pixel** (to be reconstructed)
   * Input to the model:

     ```python
     target_img = img_tensor
     masked_img = target_img * (1.0 - mask_tensor)  # zero-out masked regions
     ```

4. **Dataset**: `DentalMIMReconstructionDataset`

   Returns for each `__getitem__`:

   ```python
   masked_img  # (1, H, W), masked input
   target_img  # (1, H, W), full augmented image (reconstruction target)
   mask_tensor # (1, H, W), 1 for masked pixels
   meta        # dict with path, age_group, label_status, pair_id, ...
   ```

5. **DataLoader**:

   ```python
   mim_loader = DataLoader(
       mim_dataset,
       batch_size=BATCH_SIZE,
       shuffle=True,
       num_workers=0,  # recommended on Windows to avoid Dataloader deadlocks
       pin_memory=(device.type == "cuda"),
   )
   ```

> **Note (Windows)**: set `num_workers = 0` to avoid the DataLoader hanging at the first batch.

---

### 3.2 2D U-Net Backbone

`UNet2D` is a standard 2D U-Net:

* Input: `1 × 320 × 640` grayscale image
* Encoder: 4 downsampling stages (with `DoubleConv` blocks)
* Decoder: 4 upsampling stages with skip connections
* Output: `1 × 320 × 640` reconstructed grayscale image

Key components:

* `DoubleConv` – `(Conv2d → BN → ReLU) × 2`
* `Down` – `MaxPool2d(2) + DoubleConv`
* `Up` – `Upsample (bilinear) + concat skip + DoubleConv`
* `OutConv` – final `Conv2d` to map to 1 output channel

> This same backbone can later be reused for **segmentation** by replacing the final output layer (e.g. `out_channels = 1` or `= num_classes`) and training with segmentation losses (Dice, BCE, etc.) on labeled image–mask pairs.

---

### 3.3 MIM Loss & Training

Loss function: **masked L1 loss**, computed **only on masked pixels**:

```python
def masked_l1_loss(pred, target, mask):
    l1 = torch.abs(pred - target)     # (B, 1, H, W)
    masked_l1 = l1 * mask             # keep only masked pixels
    denom = mask.sum()
    if denom.item() < 1.0:
        return l1.mean()              # fallback if mask is empty
    return masked_l1.sum() / denom
```

Training loop (simplified):

```python
model.train()

for epoch in range(1, NUM_EPOCHS + 1):
    running_loss = 0.0

    for batch in mim_loader:
        masked_img, target_img, mask_tensor, meta = batch
        masked_img = masked_img.to(device)
        target_img = target_img.to(device)
        mask_tensor = mask_tensor.to(device)

        optimizer.zero_grad()
        pred = model(masked_img)
        loss = masked_l1_loss(pred, target_img, mask_tensor)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(mim_loader)
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Avg MIM loss: {avg_loss:.6f}")
```

**Optimizer**: typically `Adam(lr=1e-3)` is used as a starting point.

---

### 3.4 TensorBoard Logging (Optional but Recommended)

`MIM_Pretrain.ipynb` can log to TensorBoard via `SummaryWriter`:

* Scalars:

  * `train/batch_loss` – per-batch masked L1 loss
  * `train/epoch_loss` – per-epoch average loss
* Images (e.g. every N steps):

  * `mim/input_masked`   – masked input image
  * `mim/target_full`    – full augmented image
  * `mim/recon_pred`     – model reconstruction
  * `mim/abs_error`      – absolute reconstruction error

Run TensorBoard from the terminal (example):

```bash
conda activate dlcv   # or your environment name
tensorboard --logdir "E:\...\sts_tooth_data\runs"
```

Then open:

```text
http://localhost:6006
```

---

### 3.5 Saving Pretrained Weights

At the end of training:

```python
SAVE_DIR = DATA_ROOT / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

save_path = SAVE_DIR / "unet2d_mim_pretrained.pth"
torch.save(model.state_dict(), save_path)

print("Pretrained U-Net weights saved to:", save_path)
```

Later, for segmentation:

```python
model = UNet2D(in_channels=1, out_channels=NUM_CLASSES, base_channels=32)
state_dict = torch.load(save_path, map_location="cpu")

# Option 1: allow mismatch on final layer if you change out_channels
model.load_state_dict(state_dict, strict=False)
```

---

## 4. Environment Setup

A typical environment (conda) might look like:

```bash
conda create -n toothmim python=3.10 -y
conda activate toothmim

# Core scientific stack
conda install -c conda-forge numpy scipy pandas matplotlib scikit-image -y

# Deep learning (adjust CUDA/CPU as needed)
pip install torch torchvision torchaudio  # or from pytorch.org selector

# Image processing / augmentation
conda install -c conda-forge opencv albumentations -y

# Misc
pip install pillow tqdm tensorboard
```

On **Windows**, ensure:

* `num_workers = 0` in `DataLoader` to avoid deadlocks.
* Numpy / OpenCV / Albumentations versions are compatible (use conda-forge).

---

## 5. Next Steps: Segmentation Fine-Tuning (Not in This README’s Code Yet)

Once pretraining is done, the natural next step (not fully implemented here yet) is:

1. Build a `DentalSegmentationDataset` that returns `(image, mask)` pairs from:

   * `processed_2d/.../images/*.png`
   * `processed_2d/.../masks/*.png`

2. Initialize a `UNet2D` with `out_channels = 1` (binary mask) or more for multi-class:

   ```python
   seg_model = UNet2D(in_channels=1, out_channels=1, base_channels=32)
   seg_model.load_state_dict(torch.load(".../unet2d_mim_pretrained.pth"), strict=False)
   ```

3. Train the segmentation model with appropriate losses (e.g. BCE + Dice) on the ~900 labeled image–mask pairs.

---

## 6. Gotchas & Tips

* **Image resolution**:
  MIM pipeline assumes images are resized to `320 × 640`. Adjust `TARGET_HEIGHT` / `TARGET_WIDTH` consistently if you change this.

* **Masking ratio**:
  Controlled by `num_blocks` and `min_block_fraction` / `max_block_fraction`. You can tune these to make the task easier or harder.

* **Binary compatibility errors** (e.g. `numpy.dtype size changed`):
  Typically indicates mismatched versions of `numpy` and compiled libraries (OpenCV, etc.). Reinstall via conda with compatible versions.

* **Reproducibility**:
  Seeds are set via `random.seed`, `np.random.seed`, `torch.manual_seed`, and `torch.cuda.manual_seed_all` in the notebooks.

---

If you extend this project (e.g. add segmentation training notebooks, evaluation scripts, or model export), consider updating this README with the new components and usage examples.

```
::contentReference[oaicite:0]{index=0}
```
