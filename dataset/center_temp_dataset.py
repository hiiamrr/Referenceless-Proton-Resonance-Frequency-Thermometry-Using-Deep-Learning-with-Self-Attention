import torch
from torch.utils.data import Dataset
import numpy as np

def create_circular_roi_mask(shape, center, radius):
    h, w = shape
    Y, X = np.ogrid[:h, :w]
    cy, cx = center
    mask = (Y - cy)**2 + (X - cx)**2 <= radius**2
    return mask.astype(np.uint8)

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

        h, w = input_patch.shape
        cy, cx = h // 2, w // 2
        roi_mask = create_circular_roi_mask((h, w), (cy, cx), self.roi_radius)

        input_tensor = self._to_tensor(input_patch)
        target_tensor = self._to_tensor(target_patch)
        mask_tensor = torch.from_numpy(roi_mask[None]).float()  # shape [1, H, W]

        return input_tensor, target_tensor, mask_tensor

    def _to_tensor(self, x):
        real = torch.from_numpy(np.real(x)).float()
        imag = torch.from_numpy(np.imag(x)).float()
        return torch.stack([real, imag], dim=0)
