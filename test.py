import torch
import torch.nn as nn
import numpy as np
from utils.dataset import loaddata
from utils.model import Refless_Model
from utils.tvt_test import *
import os 

def prepare_dir(ndir):
    is_exists = os.path.exists(ndir)
    if not is_exists:
        os.makedirs(ndir)

device = torch.device('cuda')
test_dir = r'C:\Users\YZhao\Desktop\001-046-P+TA-Ax-1S-SPGR - Copy'
test_data = loaddata(test_dir, device)

model = Refless_Model().cuda()
criterion = nn.MSELoss()
weights_dir = r"D:\YueranZhao\DL_CFE\Pth\Referenceless_PRF_Model\test_result\weights\mixed_train_ablation\mixed_train_64_9\epoch_431_11554.750332.pt"
result_dir = r"D:\YueranZhao\DL_CFE\Pth\Referenceless_PRF_Model\test_result\result_image\sag"
prepare_dir(result_dir)

model.load_state_dict(torch.load(weights_dir))

test_err, ComplexImages, InputImages, GTImages = test_epoch(test_data, 9, 9, 64, 2, device, model, criterion)

ComplexImages_cpu = [complex_image.cpu().numpy() for complex_image in ComplexImages]

InputImages_cpu = [input_image.cpu().numpy() for input_image in InputImages]

GTImages_cpu = [gt_image.cpu().numpy() for gt_image in GTImages]

for i, complex_image in enumerate(ComplexImages_cpu):
    np.save(os.path.join(result_dir, f'complex_image_{i}.npy'),(complex_image[0, ...] + 1j * complex_image[1, ...]))

for i, input_image in enumerate(InputImages_cpu):
    np.save(os.path.join(result_dir, f'input_image_{i}.npy'), input_image[0, ...] + 1j * input_image[1, ...])

for i, gt_image in enumerate(GTImages_cpu):
    np.save(os.path.join(result_dir, f'gt_image_{i}.npy'), gt_image[0, ...] + 1j * gt_image[1, ...])
