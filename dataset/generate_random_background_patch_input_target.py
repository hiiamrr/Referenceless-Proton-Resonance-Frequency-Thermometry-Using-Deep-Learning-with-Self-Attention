import numpy as np
from scipy.ndimage import binary_erosion
import random

def get_valid_tissue_mask(mag_image: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    return (mag_image > threshold * mag_image.max()).astype(np.uint8)

def extract_patch(image: np.ndarray, center: tuple, patch_size: int = 32) -> np.ndarray:
    h, w = image.shape
    cy, cx = int(center[0]), int(center[1])
    half = patch_size // 2
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
        return None
    return image[y1:y2, x1:x2]

def apply_circular_mask(image: np.ndarray, center: tuple, radius: int) -> np.ndarray:
    h, w = image.shape
    Y, X = np.ogrid[:h, :w]
    cy, cx = center
    mask = (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2
    masked = image.copy()
    masked[mask] = 0
    return masked

def generate_random_background_patch_input_target(
    current_image: np.ndarray,
    roi_radius: int = 9,
    patch_size: int = 32,
    tissue_thresh: float = 0.05,
    center_exclude_radius: int = 20,
    max_tries: int = 100
):
    """
    Sample non-heated region patches from the current image
    """
    mag_img = np.abs(current_image)
    valid_mask = get_valid_tissue_mask(mag_img, threshold=tissue_thresh)

    h, w = current_image.shape
    if patch_size == 256:
        # full fov
        input_patch = current_image.copy()
        target_patch = current_image.copy()

        roi_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
        Yp, Xp = np.ogrid[:patch_size, :patch_size]
        center = (patch_size // 2, patch_size // 2)
        mask = (Yp - center[0]) ** 2 + (Xp - center[1]) ** 2 <= roi_radius ** 2
        roi_mask[mask] = 1

        input_patch[roi_mask == 1] = 0

        return input_patch.astype(np.complex64), target_patch.astype(np.complex64), roi_mask

    Y, X = np.ogrid[:h, :w]
    cy_full, cx_full = h // 2, w // 2
    center_mask = (Y - cy_full) ** 2 + (X - cx_full) ** 2 > center_exclude_radius**2
    valid_mask = valid_mask * center_mask

    ys, xs = np.nonzero(valid_mask)
    if len(ys) == 0:
        return None, None, None

    half = patch_size // 2

    for _ in range(max_tries):
        idx = random.randint(0, len(ys) - 1)
        cy, cx = ys[idx], xs[idx]
        if (cy - half < 0 or cy + half > h or 
            cx - half < 0 or cx + half > w):
            continue

        input_patch = extract_patch(current_image, (cy, cx), patch_size)
        target_patch = extract_patch(current_image, (cy, cx), patch_size)
        if input_patch is None or target_patch is None:
            continue

        patch_valid = get_valid_tissue_mask(np.abs(input_patch), threshold=tissue_thresh)
        if patch_valid.mean() < 0.7:
            continue  

        input_patch = apply_circular_mask(input_patch, (patch_size // 2, patch_size // 2), roi_radius)
        roi_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
        Yp, Xp = np.ogrid[:patch_size, :patch_size]
        mask = (Yp - patch_size // 2) ** 2 + (Xp - patch_size // 2) ** 2 <= roi_radius ** 2
        roi_mask[mask] = 1

        return input_patch.astype(np.complex64), target_patch.astype(np.complex64), roi_mask

    return None, None, None
