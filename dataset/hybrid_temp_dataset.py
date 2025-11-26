import torch
from torch.utils.data import Dataset
import numpy as np

class HybridTempDataset(Dataset):
    def __init__(self, center_dataset, background_dataset, center_ratio=0.7):
        """
        Combine the center-heated and background ROI datasets to form a unified training set

        - center_dataset: input, target, mask
        - background_dataset: input, target, mask
        """
        self.center_dataset = center_dataset
        self.background_dataset = background_dataset
        self.center_ratio = center_ratio

        self.center_len = len(center_dataset)
        self.background_len = 9999999
        self.max_len = max(self.center_len, 1000)  

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        use_center = torch.rand(1).item() < self.center_ratio

        if use_center and idx < self.center_len:
            return self.center_dataset[idx]
        else:
            bg_idx = torch.randint(0, len(self.background_dataset), (1,)).item()
            return self.background_dataset[bg_idx]
        
class CenterTempDataset(Dataset):
    def __init__(self, inputs, targets, roi_radius=9):
        self.inputs = inputs
        self.targets = targets
        self.roi_radius = roi_radius

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_patch = self.inputs[idx]
        target_patch = self.targets[idx]

        mask = np.zeros_like(np.real(input_patch), dtype=np.uint8)
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        roi_mask = ((Y - cy)**2 + (X - cx)**2 <= self.roi_radius**2).astype(np.uint8)

        real = torch.from_numpy(np.real(input_patch)).float()
        imag = torch.from_numpy(np.imag(input_patch)).float()
        target_real = torch.from_numpy(np.real(target_patch)).float()
        target_imag = torch.from_numpy(np.imag(target_patch)).float()
        mask_tensor = torch.from_numpy(roi_mask[None]).float()

        return torch.stack([real, imag]), torch.stack([target_real, target_imag]), mask_tensor
