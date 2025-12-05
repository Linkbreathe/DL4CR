# ToothSeg: Self-Supervised Pretraining & Segmentation for Dental X-rays

This repository implements a framework for dental X-ray segmentation, featuring self-supervised pretraining (MIM & Full Image) and supervised fine-tuning.

## Dataset
>TG3K: The TG3K dataset is obtained from 16 ultrasound videos. Initially, the dataset was designed for the accurate segmentation of the thyroid gland area from videos. The extraction process first involved extracting frames from videos. To ensure the quality of the dataset, a rule was established: only images where the thyroid gland area occupies more than 0.06 of the total image area were retained. Consequently, the TG3K dataset consists of such high-quality images, including 3583 ultrasound images. The unique aspect of this dataset is its use for the segmentation of thyroid nodules, a valuable and challenging task. The TG3K dataset is designed for the segmentation of thyroid nodules, a task that is both clinically significant and challenging. Accurate segmentation of thyroid nodules is crucial for the early diagnosis and treatment of thyroid cancer. Through in-depth research on this dataset, it is possible to better understand and address the challenges in thyroid nodule segmentation, contributing significantly to improving the diagnostic accuracy and treatment efficacy of thyroid cancer.

Download Link: https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TG3K.md

STS_Tooth_Data:
>In response to the increasing prevalence of dental diseases, dental health, a vital aspect of human well-being, warrants greater attention. Panoramic X-ray images (PXI) and Cone Beam Computed Tomography (CBCT) are key tools for dentists in diagnosing and treating dental conditions. Additionally, deep learning for tooth segmentation can focus on relevant treatment information and localize lesions. However, the scarcity of publicly available PXI and CBCT datasets hampers their use in tooth segmentation tasks. Therefore, this paper presents a multimodal dataset for semi-supervised deep learning in dental PXI and CBCT, named STS-2D-Tooth and STS-3D-Tooth. STS-2D-Tooth includes 4,000 images and 900 masks, categorized by age into children and adults. Moreover, we have collected CBCTs providing more detailed and three-dimensional information, resulting in the STS-3D-Tooth dataset comprising 148,400 unlabeled scans and 8,800 masks. To our knowledge, this is the first multimodal dataset combining dental PXI and CBCT, and it is the largest tooth segmentation dataset, a significant step forward for the advancement of tooth segmentation.

Download Link: https://doi.org/10.5281/zenodo.10597292


## Core Scripts & Modules

This project is built around the following key files:

### 1. Training & Evaluation

#### `tg3k_MIM_Pretrain.py`
**Description:** Implements **Masked Image Modeling (MIM)** pretraining. It trains a U-Net backbone to reconstruct masked regions of input images, learning robust feature representations from unlabeled data.
**Usage:**
```bash
python tg3k_MIM_Pretrain.py \
  --images-dir ./sts_tooth_data/processed_2d \
  --checkpoint-dir ./tg3k/checkpoints_mim/unet \
  --model-type unet \
  --mask-ratio 0.55
```

#### `tg3k_full_image_pretrain.py`
**Description:** Implements **Full Image Reconstruction** pretraining. It trains the model to reconstruct the original image from a perturbed version (e.g., with noise or augmentation), serving as an alternative self-supervised strategy.
**Usage:**
```bash
python tg3k_full_image_pretrain.py \
  --images-dir ./sts_tooth_data/processed_2d \
  --checkpoint-dir ./tg3k/checkpoints_fullimage/unet \
  --model-type unet
```

#### `tg3k_finetune_segementation.py`
**Description:** The main script for **Segmentation Fine-tuning**. It loads a pretrained backbone (from MIM or Full Image pretraining) and trains the model on labeled image-mask pairs using a combined BCE + Dice loss.
**Usage:**
```bash
python tg3k_finetune_segementation.py \
  --images-dir ./sts_tooth_data/processed_2d \
  --masks-dir ./sts_tooth_data/processed_2d \
  --init-from ./tg3k/checkpoints_mim/unet/mim_best.pth \
  --save-dir ./tg3k/checkpoints_finetune/mim_unet \
  --model-type unet
```

#### `tg3k_eval.py`
**Description:** Evaluates a trained segmentation model on a specific dataset split (val or test). It computes quantitative metrics such as **Dice Score** and **IoU**.
**Usage:**
```bash
python tg3k_eval.py \
  --images-dir ./sts_tooth_data/processed_2d \
  --masks-dir ./sts_tooth_data/processed_2d \
  --checkpoint ./tg3k/checkpoints_finetune/mim_unet/best_model.pth \
  --split test
```

### 2. Models & Utilities

#### `models.py`
**Description:** Defines the PyTorch model architectures used throughout the project:
*   `UNet2D`: Standard U-Net.
*   `UNet2D_scAG`: U-Net with Spatial-Channel Attention Gates.
*   `UNet2D_NAC`: U-Net with Neighbor-Aware Context blocks.

#### `utils.py`
**Description:** Contains essential utility functions shared across scripts, including:
*   `TG3KMaskedReconstructionDataset`: Dataset class for MIM.
*   `build_ultrasound_transform`: Albumentations transform builder.
*   `set_seed`: For reproducibility.
*   `save_checkpoint` / `load_checkpoint`: Checkpoint management.

#### `tool.py`
**Description:** A collection of miscellaneous utility snippets. Useful for inspecting checkpoint files (e.g., printing keys in a state dict) or quick debugging tasks.

#### `fbf.py`
**Description:** A preprocessing script that applies a **Fast Bilateral Filter (FBF)** to images. This reduces noise while preserving edges, potentially improving downstream segmentation performance.
**Usage:**
```bash
python fbf.py
```

### 3. Visualization

#### `visualizer.py`
**Description:** Provides helper functions for visualizing results, such as:
*   `visualize_samples_in_one_plot`: Plots input, ground truth, prediction, and overlay side-by-side.
*   `get_saliency_map`: Generates gradient-based saliency maps to visualize model focus.

#### `visual_helper.ipynb`
**Description:** An interactive Jupyter notebook for:
*   Qualitative analysis of model predictions.
*   Comparing different models (e.g., MIM vs. Full Image) on the same samples.
*   Visualizing best and worst performing cases.

---

## Quick Start Workflow

1.  **Preprocess (Optional):** Run `python fbf.py` to denoise images.
2.  **Pretrain:** Run `tg3k_MIM_Pretrain.py` (or `tg3k_full_image_pretrain.py`) on your dataset.
3.  **Fine-tune:** Run `tg3k_finetune_segementation.py` using the best checkpoint from step 2.
4.  **Evaluate:** Run `tg3k_eval.py` to get test metrics.
5.  **Visualize:** Open `visual_helper.ipynb` to inspect the results visually.
