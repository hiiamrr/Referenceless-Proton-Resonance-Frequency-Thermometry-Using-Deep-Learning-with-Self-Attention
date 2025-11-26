import torch
from torch.utils.data import Dataset
import numpy as np
import random
from dataset.generate_random_background_patch_input_target import generate_random_background_patch_input_target

class EdgeROIVerificationDataset(Dataset):
    def __init__(self, raw_images: list[np.ndarray],
                 patch_size: int = 64,
                 roi_radius: int = 9,
                 tissue_thresh: float = 0.05,
                 center_exclude_radius: int = 20,
                 n_samples: int = 200):
        
        self.samples = []
        tries = 0
        while len(self.samples) < n_samples and tries < n_samples * 10:
            img = random.choice(raw_images)
            input_patch, target_patch, roi_mask = generate_random_background_patch_input_target(
                img,
                patch_size=patch_size,
                roi_radius=roi_radius,
                tissue_thresh=tissue_thresh,
                center_exclude_radius=center_exclude_radius
            )
            if input_patch is not None:
                self.samples.append((
                    input_patch.astype(np.complex64),
                    target_patch.astype(np.complex64),
                    roi_mask.astype(np.uint8)
                ))
            tries += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_patch, target_patch, roi_mask = self.samples[idx]
        return (
            self._to_tensor(input_patch),
            self._to_tensor(target_patch),
            torch.from_numpy(roi_mask[None]).float()
        )

    def _to_tensor(self, x):
        real = torch.from_numpy(np.real(x)).float()
        imag = torch.from_numpy(np.imag(x)).float()
        return torch.stack([real, imag], dim=0)


def validate_edge(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, masks in dataloader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            outputs = model(inputs, masks)
            loss = criterion(outputs, targets, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)
