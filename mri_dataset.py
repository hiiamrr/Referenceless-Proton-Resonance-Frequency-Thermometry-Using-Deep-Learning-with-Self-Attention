import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MRITemperatureDataset(Dataset):
    def __init__(self, data_root, split='train', split_ratio=(0.85, 0.15), seed=42):
        self.input_dir = os.path.join(data_root, 'input')
        self.target_dir = os.path.join(data_root, 'target')
        self.mask_dir = os.path.join(data_root, 'mask')

        self.file_names = sorted(os.listdir(self.input_dir))
        np.random.seed(seed)
        np.random.shuffle(self.file_names)

        total = len(self.file_names)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])

        if split == 'train':
            self.file_names = self.file_names[:train_end]
        elif split == 'val':
            self.file_names = self.file_names[train_end:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        input_img = np.load(os.path.join(self.input_dir, name))
        target_img = np.load(os.path.join(self.target_dir, name))
        mask = np.load(os.path.join(self.mask_dir, name))
        # print("input shape", input_img.shape)

        # Convert to tensors
        # 分别将实部、虚部分开作为两个通道
        input_tensor = torch.from_numpy(np.stack([input_img.real, input_img.imag], axis=0)).float()
        target_tensor = torch.from_numpy(np.stack([target_img.real, target_img.imag], axis=0)).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return input_tensor, target_tensor, mask_tensor
