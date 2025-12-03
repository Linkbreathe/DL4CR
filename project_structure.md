# Project Structure

This document outlines the directory structure and the purpose of key files in the project.

## Root Directory

| File / Directory | Description |
| :--- | :--- |
| **`tg3k_MIM_Pretrain.py`** | Script for **Masked Image Modeling (MIM)** pretraining on the TG3K dataset. Implements self-supervised learning by masking parts of the image and reconstructing them. |
| **`tg3k_finetune_segementation.py`** | Script for **fine-tuning** the segmentation model. Loads pretrained weights (from MIM or other sources) and trains on labeled segmentation data. |
| **`tg3k_full_image_pretrain.py`** | Alternative pretraining script using full image reconstruction or auto-encoding tasks. |
| **`tg3k_eval.py`** | Evaluation script for testing trained models and calculating metrics (e.g., Dice score). |
| **`models.py`** | Contains model architecture definitions (e.g., U-Net, encoders, decoders). |
| **`utils.py`** | Common utility functions used across multiple scripts (e.g., data loading helpers, metrics). |
| **`visualizer.py`** | Helper script for visualizing images, masks, and model predictions. |
| **`tool.py`** | Miscellaneous utility tools. |
| **`visual_helper.ipynb`** | Jupyter notebook for interactive visualization and debugging of data/results. |
| **`environment.yml`** | Conda environment configuration file listing dependencies. |

## Data & Resources

| Directory | Description |
| :--- | :--- |
| **`tg3k/`** | Main directory for the **TG3K** dataset (images, masks, splits). |
| **`Dental/`** | Directory containing Dental dataset resources or related code. |
| **`sts_tooth_data/`** | Directory for STS tooth dataset. |
| **`train_finetune_script/`** | Likely contains shell scripts or configs for running fine-tuning experiments. |

## Key Workflows

1.  **Pretraining**: Run `tg3k_MIM_Pretrain.py` to train the encoder on unlabeled data.
2.  **Fine-tuning**: Run `tg3k_finetune_segementation.py` using the pretrained weights to train the segmentation head.
3.  **Evaluation**: Run `tg3k_eval.py` to assess model performance on the test set.
