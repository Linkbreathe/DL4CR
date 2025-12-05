# =========================
# Cell 1: Basic Configuration & Imports
# =========================
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Input root directory: As specified, ./tg3k/tg3k
ROOT_DIR = Path("./tg3k/tg3k").resolve()
IMAGE_DIR = ROOT_DIR / "thyroid-image"
MASK_DIR = ROOT_DIR / "thyroid-mask"
JSON_PATH = ROOT_DIR / "tg3k-trainval.json"

# Output root directory: Do not overwrite original data, write to tg3k_fbf
OUT_ROOT = ROOT_DIR.parent / "tg3k_fbf"
OUT_IMAGE_DIR = OUT_ROOT / "thyroid-image"
OUT_MASK_DIR = OUT_ROOT / "thyroid-mask"

print("Input root:", ROOT_DIR)
print("Output root:", OUT_ROOT)


# =========================
# Cell 2: Fast Bilateral Filter Implementation (Grayscale)
# =========================
def fast_bilateral_filter_gray(
    img: np.ndarray,
    downsample_ratio: int = 2,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 15.0,
) -> np.ndarray:
    """
    Applies an approximate Fast Bilateral Filter to a single-channel grayscale image.

    Args:
        img: np.ndarray, shape (H, W), dtype uint8 or float32.
        downsample_ratio: Downsampling factor to accelerate bilateral filtering.
        d: Neighborhood diameter (passed to cv2.bilateralFilter).
        sigma_color: Sigma for color/intensity difference (passed to cv2.bilateralFilter).
        sigma_space: Sigma for spatial distance (passed to cv2.bilateralFilter).

    Returns:
        Processed grayscale image, shape (H, W), dtype uint8.
    """
    if img.ndim != 2:
        raise ValueError(f"fast_bilateral_filter_gray expects 2D array, got {img.shape}")

    # Convert to 8-bit grayscale for easier processing
    if img.dtype != np.uint8:
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_8u = img

    h, w = img_8u.shape

    # Downsample for acceleration
    if downsample_ratio > 1:
        small = cv2.resize(
            img_8u,
            (w // downsample_ratio, h // downsample_ratio),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img_8u

    # Bilateral filter (edge-preserving)
    small_filtered = cv2.bilateralFilter(
        small,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    # Upsample back to original resolution
    if downsample_ratio > 1:
        filtered_8u = cv2.resize(
            small_filtered,
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        filtered_8u = small_filtered

    return filtered_8u


# =========================
# Cell 3: Prepare output directories, copy masks & json
# =========================
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 1) Copy masks (keep original masks, do not apply FBF)
if MASK_DIR.exists():
    if not OUT_MASK_DIR.exists():
        print("Copying masks to", OUT_MASK_DIR)
        shutil.copytree(MASK_DIR, OUT_MASK_DIR)
    else:
        print("Mask dir already exists, skip copy:", OUT_MASK_DIR)
else:
    print("WARNING: mask dir not found:", MASK_DIR)

# 2) Copy split json
if JSON_PATH.exists():
    out_json = OUT_ROOT / JSON_PATH.name
    if not out_json.exists():
        print("Copying split json to", out_json)
        shutil.copy2(JSON_PATH, out_json)
    else:
        print("Split json already exists, skip copy:", out_json)
else:
    print("WARNING: split json not found:", JSON_PATH)


# =========================
# Cell 4: Iterate over all images and apply FBF
# =========================
# Supported image extensions
EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

image_paths = []
for ext in EXTS:
    image_paths.extend(IMAGE_DIR.rglob(f"*{ext}"))
image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

for img_path in tqdm(image_paths):
    # Read grayscale image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read:", img_path)
        continue

    # Apply FBF
    img_fbf = fast_bilateral_filter_gray(img)

    # Output path: Maintain relative directory structure
    rel_path = img_path.relative_to(IMAGE_DIR)
    out_path = OUT_IMAGE_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    ok = cv2.imwrite(str(out_path), img_fbf)
    if not ok:
        print("Failed to write:", out_path)

print("Done! FBF-processed images saved to:", OUT_IMAGE_DIR)