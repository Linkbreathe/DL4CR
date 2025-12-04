# =========================
# Cell 1: 基本配置 & 导入
# =========================
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# 输入根目录：你说的是 ./tg3k/tg3k
ROOT_DIR = Path("./tg3k/tg3k").resolve()
IMAGE_DIR = ROOT_DIR / "thyroid-image"
MASK_DIR = ROOT_DIR / "thyroid-mask"
JSON_PATH = ROOT_DIR / "tg3k-trainval.json"

# 输出根目录：不覆盖原数据，写到 tg3k_fbf
OUT_ROOT = ROOT_DIR.parent / "tg3k_fbf"
OUT_IMAGE_DIR = OUT_ROOT / "thyroid-image"
OUT_MASK_DIR = OUT_ROOT / "thyroid-mask"

print("Input root:", ROOT_DIR)
print("Output root:", OUT_ROOT)


# =========================
# Cell 2: Fast Bilateral Filter 实现（灰度图）
# =========================
def fast_bilateral_filter_gray(
    img: np.ndarray,
    downsample_ratio: int = 2,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 15.0,
) -> np.ndarray:
    """
    对单通道灰度图应用近似的 Fast Bilateral Filter。

    Args:
        img: np.ndarray, shape (H, W)，dtype uint8 或 float32。
        downsample_ratio: 下采样倍数，用于加速双边滤波。
        d: 邻域直径（传给 cv2.bilateralFilter）。
        sigma_color: 灰度差的 sigma（传给 cv2.bilateralFilter）。
        sigma_space: 空间距离的 sigma（传给 cv2.bilateralFilter）。

    Returns:
        处理后的灰度图，shape (H, W)，dtype uint8。
    """
    if img.ndim != 2:
        raise ValueError(f"fast_bilateral_filter_gray expects 2D array, got {img.shape}")

    # 统一转成 8-bit 灰度方便处理
    if img.dtype != np.uint8:
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_8u = img

    h, w = img_8u.shape

    # 下采样，加速
    if downsample_ratio > 1:
        small = cv2.resize(
            img_8u,
            (w // downsample_ratio, h // downsample_ratio),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img_8u

    # 双边滤波（edge-preserving）
    small_filtered = cv2.bilateralFilter(
        small,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    # 上采样回原分辨率
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
# Cell 3: 准备输出目录，复制 mask & json
# =========================
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 1) 复制 mask（保持原始 mask，不做 FBF）
if MASK_DIR.exists():
    if not OUT_MASK_DIR.exists():
        print("Copying masks to", OUT_MASK_DIR)
        shutil.copytree(MASK_DIR, OUT_MASK_DIR)
    else:
        print("Mask dir already exists, skip copy:", OUT_MASK_DIR)
else:
    print("WARNING: mask dir not found:", MASK_DIR)

# 2) 复制 split json
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
# Cell 4: 遍历所有图像并应用 FBF
# =========================
# 支持的图像后缀
EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

image_paths = []
for ext in EXTS:
    image_paths.extend(IMAGE_DIR.rglob(f"*{ext}"))
image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

for img_path in tqdm(image_paths):
    # 读灰度图
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read:", img_path)
        continue

    # 应用 FBF
    img_fbf = fast_bilateral_filter_gray(img)

    # 输出路径：保持相对目录结构
    rel_path = img_path.relative_to(IMAGE_DIR)
    out_path = OUT_IMAGE_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存
    ok = cv2.imwrite(str(out_path), img_fbf)
    if not ok:
        print("Failed to write:", out_path)

print("Done! FBF-processed images saved to:", OUT_IMAGE_DIR)
