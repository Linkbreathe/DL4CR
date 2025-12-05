# Project Structure

This document outlines the directory structure and the purpose of key files in the **ToothSeg** project.

## 📂 Root Directory

| File / Directory | Description |
| :--- | :--- |
| **`tg3k_MIM_Pretrain.py`** | **[Pretraining]** Script for **Masked Image Modeling (MIM)**. Trains a U-Net to reconstruct masked regions of input images. |
| **`tg3k_full_image_pretrain.py`** | **[Pretraining]** Script for **Full Image Reconstruction**. Trains a U-Net to reconstruct the original image from a perturbed version (e.g., noisy/augmented). |
| **`tg3k_finetune_segementation.py`** | **[Fine-tuning]** Script for **Segmentation Fine-tuning**. Loads pretrained weights (MIM or Full Image) and trains on labeled image-mask pairs. |
| **`tg3k_eval.py`** | **[Evaluation]** Script for evaluating trained models on test sets. Calculates metrics like **Dice Score** and **IoU**. |
| **`models.py`** | **[Core]** Defines PyTorch model architectures: `UNet2D`, `UNet2D_scAG`, `UNet2D_NAC`. |
| **`utils.py`** | **[Core]** Common utilities: data loading, random seeding, Albumentations transforms, checkpoint management. |
| **`visualizer.py`** | **[Visualization]** Helper functions for plotting images, masks, overlays, and saliency maps. |
| **`fbf.py`** | **[Preprocessing]** Script to apply **Fast Bilateral Filter (FBF)** to images for noise reduction. |
| **`visual_helper.ipynb`** | **[Visualization]** Jupyter notebook for interactive visualization, model comparison, and error analysis. |
| **`tool.py`** | Miscellaneous utility snippets (e.g., inspecting checkpoint keys). |
| **`environment.yml`** | Conda environment configuration file. |
| **`README.md`** | Main project documentation. |

---

## 📂 Subdirectories

### `Tooth/` (Data Preparation)
Contains notebooks for downloading and preparing the **STS-2D-Tooth** dataset.

| File | Description |
| :--- | :--- |
| **`PreProcessing.ipynb`** | Downloads raw data from Zenodo, merges zip parts, extracts them, and organizes the directory structure. |
| **`Data_enchance.ipynb`** | Analyzes dataset statistics, creates train/val/test splits, and visualizes data augmentations. |
| **`MIM_Pretrain.ipynb`** | (Legacy) Notebook version of MIM pretraining. |
| **`Full_Image_Pretrain.ipynb`** | (Legacy) Notebook version of Full Image pretraining. |
| **`Fine_tuning.ipynb`** | (Legacy) Notebook version of segmentation fine-tuning. |

### `sts_tooth_data/` (Dataset)
The main data directory (created by `PreProcessing.ipynb`).

| Directory / File | Description |
| :--- | :--- |
| **`processed_2d/`** | Contains the organized dataset: `adult/`, `children/`, `labeled/`, `unlabeled/`. |
| **`sts2d_index.csv`** | Master CSV index of all images and their metadata (age group, label status, pair ID). |
| **`raw/`** | Extracted raw data from the downloaded zip. |
| **`downloads/`** | Raw downloaded zip parts. |

### `tg3k/` (Experiments & Logs)
Default directory for saving checkpoints and logs.

| Directory | Description |
| :--- | :--- |
| **`checkpoints_mim/`** | Saved checkpoints from MIM pretraining. |
| **`checkpoints_fullimage/`** | Saved checkpoints from Full Image pretraining. |
| **`checkpoints_finetune/`** | Saved checkpoints from segmentation fine-tuning. |
| **`runs/`** | TensorBoard logs for all experiments. |
| **`tg3k/`** | (Legacy) Original dataset folder structure (if used). |

### `tg3k_fbf/` (Filtered Data)
Optional directory created by `fbf.py` containing bilateral-filtered images.

---

## 🔄 Key Workflows

1.  **Data Prep**: `Tooth/PreProcessing.ipynb` → `Tooth/Data_enchance.ipynb`
2.  **Pretraining**: `tg3k_MIM_Pretrain.py` **OR** `tg3k_full_image_pretrain.py`
3.  **Fine-tuning**: `tg3k_finetune_segementation.py` (loads weights from step 2)
4.  **Evaluation**: `tg3k_eval.py`
5.  **Visualization**: `visual_helper.ipynb`
