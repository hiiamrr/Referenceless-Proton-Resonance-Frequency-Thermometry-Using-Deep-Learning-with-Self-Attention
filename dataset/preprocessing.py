import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

root_path = r"C:\Users\YZhao\Desktop\Brain Dataset\mask_image"
save_root = r"C:\Users\YZhao\Desktop\ProcessedDataset_64_9"
roi_radius = 9  
crop_size = 64  
key_name = "image_seg"  

os.makedirs(os.path.join(save_root, "input"), exist_ok=True)
os.makedirs(os.path.join(save_root, "target"), exist_ok=True)
os.makedirs(os.path.join(save_root, "mask"), exist_ok=True)

def create_circular_mask(h, w, radius):
    Y, X = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8)

def crop_center(img, size):
    h, w = img.shape[-2:]
    ch, cw = size // 2, size // 2
    center_h, center_w = h // 2, w // 2
    return img[..., center_h - ch:center_h + ch, center_w - cw:center_w + cw]

sample_idx = 0
for group in tqdm(os.listdir(root_path)):
    group_path = os.path.join(root_path, group)
    if not os.path.isdir(group_path):
        continue

    mat_files = sorted([f for f in os.listdir(group_path) if f.endswith(".mat")])
    if len(mat_files) < 2:
        continue

    bg_data = sio.loadmat(os.path.join(group_path, mat_files[0]))[key_name]
    bg_img = bg_data if np.iscomplexobj(bg_data) else bg_data.astype(np.complex64)
    h, w = bg_img.shape
    mask_full = create_circular_mask(h, w, roi_radius)

    for idx, mat_file in enumerate(mat_files):
        img_data = sio.loadmat(os.path.join(group_path, mat_file))[key_name]
        img = img_data if np.iscomplexobj(img_data) else img_data.astype(np.complex64)

        # input
        input_img = img.copy()
        input_img[mask_full == 1] = 0

        # target
        if idx == 0:
            target_img = img.copy()
        else:
            target_img = img.copy()
            target_img[mask_full == 1] = bg_img[mask_full == 1]

        input_cropped = crop_center(input_img, crop_size)
        target_cropped = crop_center(target_img, crop_size)
        mask_cropped = crop_center(mask_full, crop_size)

        # save
        np.save(os.path.join(save_root, "input", f"img{sample_idx}.npy"), input_cropped)
        np.save(os.path.join(save_root, "target", f"img{sample_idx}.npy"), target_cropped)
        np.save(os.path.join(save_root, "mask", f"img{sample_idx}.npy"), mask_cropped)
        sample_idx += 1

print(f"All images have been processed. Total samples generated: {sample_idx}")
