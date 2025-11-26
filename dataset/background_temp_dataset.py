import torch
from torch.utils.data import Dataset
import numpy as np
import random
from dataset.generate_random_background_patch_input_target import generate_random_background_patch_input_target

class BackgroundTempDataset(Dataset):
    def __init__(self, raw_images, patch_size=32, roi_radius=9,
                 tissue_thresh=0.05, center_exclude_radius=20, return_mask=True):
        self.raw_images = raw_images
        self.patch_size = patch_size
        self.roi_radius = roi_radius
        self.tissue_thresh = tissue_thresh
        self.center_exclude_radius = center_exclude_radius
        self.return_mask = return_mask

    def __len__(self):
        return 999999

    def __getitem__(self, idx):
        for _ in range(10):
            img = random.choice(self.raw_images)
            input_patch, target_patch, roi_mask = generate_random_background_patch_input_target(
                img,
                roi_radius=self.roi_radius,
                patch_size=self.patch_size,
                tissue_thresh=self.tissue_thresh,
                center_exclude_radius=self.center_exclude_radius
            )
            if input_patch is not None:
                input_tensor = self._to_tensor(input_patch)
                target_tensor = self._to_tensor(target_patch)
                mask_tensor = torch.from_numpy(roi_mask[None]).float()
                return input_tensor, target_tensor, mask_tensor

        dummy = torch.zeros(2, self.patch_size, self.patch_size)
        return dummy, dummy, torch.zeros(1, self.patch_size, self.patch_size)

    def _to_tensor(self, x):
        real = torch.from_numpy(np.real(x)).float()
        imag = torch.from_numpy(np.imag(x)).float()
        return torch.stack([real, imag], dim=0)
