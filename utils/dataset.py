import os
import scipy.io
import numpy as np
import torch

def norm(x):
    x = torch.tensor(x)
    y = x / torch.max(torch.abs(x))
    return y

def loaddata(data_path, device):
    data_files = os.listdir(data_path)
    files = [os.path.join(data_path, f) for f in data_files if os.path.isfile(os.path.join(data_path, f))]
    dataset = np.empty([len(files), 256, 256], dtype='complex128')
    cfe_dataset = np.empty([len(files), 2, 256, 256], dtype=float)
    for i in range(len(files)):
        dataset_temp = scipy.io.loadmat(files[i])['image_seg']
        dataset[i, ...] = dataset_temp
        # cfe_dataset[i, 0, ...] = norm(np.real(dataset[i, ...]))
        # cfe_dataset[i, 1, ...] = norm(np.imag(dataset[i, ...]))
        cfe_dataset[i, 0, ...] = np.real(dataset[i, ...])
        cfe_dataset[i, 1, ...] = np.imag(dataset[i, ...])
    return cfe_dataset

    
