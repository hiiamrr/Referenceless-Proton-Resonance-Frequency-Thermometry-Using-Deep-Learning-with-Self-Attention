import os
import time
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os.path import join
from scipy.io import loadmat
from argparse import ArgumentParser
from yamlinclude import YamlIncludeConstructor
from utils.dataset import loaddata
from utils.model import Refless_Model
from utils.tvt import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from mri_dataset import MRITemperatureDataset
from dataset.center_temp_dataset import CenterTempDataset
from dataset.background_temp_dataset import BackgroundTempDataset
from dataset.hybrid_temp_dataset import HybridTempDataset
from dataset.edge_val_dataset import EdgeROIVerificationDataset, validate_edge

def prepare_dir(ndir):
    is_exists = os.path.exists(ndir)
    if not is_exists:
        os.makedirs(ndir)

def inherit_constructor(loader, node):
    def merge_dict(dict1, dict2):
        for key in dict2:
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dict(dict1[key], dict2[key])
            else:
                dict1[key] = dict2[key]
    
    kwargs = loader.construct_mapping(node, deep=True)
    merge = kwargs.pop("_BASE_")
    merge_dict(merge, kwargs)
    return merge

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-g', '--gpus', type=int, required=False, default=1, nargs='+', help='gpu index')
    parser.add_argument('-cd', '--cfg_dir', type=str, default='./configs', help='(str optional) the directory of config that saves other paths.')

    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k.lower()}' if parent_key else k.lower()
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def parse_args_from_dict(cfgs, parser):
        flat_cfgs = flatten_dict(cfgs)
        for key, value in flat_cfgs.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)
        return parser.parse_args(), flat_cfgs

    core_args = parser.parse_args()
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir="./configs")
    yaml.add_constructor("!inherit", inherit_constructor)

    cfgs_dir = os.path.join(core_args.cfg_dir, f"base.yaml")
    assert os.path.exists(cfgs_dir), \
        f"You try to run `base.yaml`, but there is no config file named `base.yaml` on your config directory ({core_args.cfg_dir})"
    with open(cfgs_dir) as fconfig:
        cfgs = yaml.load(fconfig.read(), Loader=yaml.FullLoader)

    args, flat_cfgs = parse_args_from_dict(cfgs, parser)

    args.model_learning_rate = flat_cfgs["model_learning_rate"] = float(args.model_learning_rate)
    args.model_weight_decay = flat_cfgs["model_weight_decay"] = float(args.model_weight_decay)

    batch_size = args.data_batch_size
    learning_rate = args.model_learning_rate
    num_epochs = args.model_num_epochs
    weight_decay = args.model_weight_decay

    gpu = args.gpus
    torch.cuda.set_device(gpu)
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0')

    class ComplexMagnitudePhaseLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pred, target, mask=None):
            pred_complex = torch.complex(pred[:, 0], pred[:, 1]) 
            target_complex = torch.complex(target[:, 0], target[:, 1])

            amplitude_error = torch.abs(torch.abs(pred_complex) - torch.abs(target_complex))
            phase_diff = torch.angle(pred_complex * torch.conj(target_complex))
            phase_error = torch.abs(torch.abs(target_complex) * phase_diff)

            total_loss = amplitude_error + 2 * phase_error

            if mask is not None:
                total_loss = total_loss * mask.squeeze(1)

            return total_loss.mean()

    def train(model, dataloader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for inputs, targets, masks in dataloader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets, masks in dataloader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
                outputs = model(inputs, masks)
                loss = criterion(outputs, targets, masks)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    model_dir = r'./Pth'
    prepare_dir(model_dir)
    model_name = r'Referenceless_PRF_Model'
    model = Refless_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.to(device)
    criterion = ComplexMagnitudePhaseLoss()
    criterion = criterion.to(device)

    pth_dir = join(model_dir, '%s' % model_name)
    log_dir = pth_dir + r'\logs\mixed_train_64_9_0.1'
    save_dir = pth_dir + r'\test_result'
    weights_dir = save_dir + r'\weights\mixed_train_ablation\mixed_train_64_9_0.1'
    # result_dir = save_dir + r'\complex_image\20250305'
    prepare_dir(pth_dir)
    prepare_dir(log_dir)
    prepare_dir(save_dir)
    prepare_dir(weights_dir)
    # prepare_dir(result_dir)
    writer = SummaryWriter(log_dir=log_dir)

    def train_dual_loader(model, center_loader, edge_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        total_batches = min(len(center_loader), len(edge_loader))

        for (input_c, target_c, mask_c), (input_e, target_e, mask_e) in zip(center_loader, edge_loader):
            inputs  = torch.cat([input_c, input_e], dim=0).to(device)
            targets = torch.cat([target_c, target_e], dim=0).to(device)
            masks   = torch.cat([mask_c, mask_e], dim=0).to(device)

            # shuffle
            perm = torch.randperm(inputs.size(0))
            inputs = inputs[perm].to(device)
            targets = targets[perm].to(device)
            masks = masks[perm].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / total_batches
    
    def remove_center_roi(image, center=None, radius=20):
        h, w = image.shape
        if center is None:
            center = (h // 2, w // 2)
        y, x = np.ogrid[:h, :w]
        mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 > radius ** 2
        return image * mask
    
    # load center dataset + train / val
    center_input_dir = r"C:\Users\YZhao\Desktop\ProcessedDataset_64_9\input"
    center_target_dir = r"C:\Users\YZhao\Desktop\ProcessedDataset_64_9\target"
    center_inputs = np.stack([np.load(os.path.join(center_input_dir, f)) for f in sorted(os.listdir(center_input_dir))])
    center_targets = np.stack([np.load(os.path.join(center_target_dir, f)) for f in sorted(os.listdir(center_target_dir))])
    num_total = center_inputs.shape[0]
    num_train = int(0.8 * num_total)
    train_center_dataset = CenterTempDataset(center_inputs[:num_train], center_targets[:num_train], roi_radius=9)
    val_center_dataset = CenterTempDataset(center_inputs[num_train:], center_targets[num_train:], roi_radius=9)

    # load raw dataset + train / val
    raw_root_dir = r"C:\Users\YZhao\Desktop\Brain Dataset\mask_image"
    raw_images = []
    for subdir in sorted(os.listdir(raw_root_dir)):
        full_subdir = os.path.join(raw_root_dir, subdir)
        if os.path.isdir(full_subdir):
            for f in sorted(os.listdir(full_subdir)):
                if f.endswith('.mat'):
                    data = loadmat(os.path.join(full_subdir, f))
                    if 'image_seg' in data:
                        raw_images.append(data['image_seg'])
    num_raw = len(raw_images)
    num_raw_train = int(0.8 * num_raw)
    train_raw_images = raw_images[:num_raw_train]
    val_raw_images = raw_images[num_raw_train:]


    # train background dataset
    train_background_dataset = BackgroundTempDataset(
        raw_images=train_raw_images,
        patch_size=64,
        roi_radius=9,
        tissue_thresh=0.05,
        center_exclude_radius=20,
        return_mask=True,
    )

    # val background dataset
    val_edge_dataset = EdgeROIVerificationDataset(
        raw_images=val_raw_images,
        patch_size=64,
        roi_radius=9,
        tissue_thresh=0.05,
        center_exclude_radius=20,
        n_samples=200
    )

    center_loader = DataLoader(train_center_dataset, batch_size=int(batch_size * 0.1), shuffle=True, drop_last=True)
    edge_loader   = DataLoader(train_background_dataset, batch_size=int(batch_size * 0.9), shuffle=True, drop_last=True)
    val_loader    = DataLoader(val_center_dataset, batch_size=batch_size, shuffle=False)
    val_edge_loader = DataLoader(val_edge_dataset, batch_size=batch_size, shuffle=False)

    model = Refless_Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = ComplexMagnitudePhaseLoss().to(device)


    # train
    min_valid_err = 1e10
    for epoch in range(num_epochs):
        train_loss = train_dual_loader(model, center_loader, edge_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        val_edge_loss = validate_edge(model, val_edge_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train: {train_loss:.3f} | Val: {val_loss:.3f} | Val Edge: {val_edge_loss:.3f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid_center', val_loss, epoch)
        writer.add_scalar('Loss/valid_edge', val_edge_loss, epoch)

        if val_loss < min_valid_err or epoch == num_epochs - 1:
            min_valid_err = val_loss
            name = f'epoch_{epoch:03d}_{val_loss*1000:.06f}.pt'
            torch.save(model.state_dict(), join(weights_dir, name))
            print(f"Model saved at {join(weights_dir, name)}")

