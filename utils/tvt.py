import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

epsilon = 1e-10
alpha = 50

def ROI_crop(image, ROI_inner, ROI_outer, input_size):
    b, c, h, w = image.shape
    ROI_cropped_image = np.copy(image)
    Input_image = np.zeros((b, c, input_size, input_size))
    Ori_image = np.zeros((b, c, input_size, input_size))
    Loss_mask = np.zeros((b, c, input_size, input_size))
    for i in range(b):
        focus_x = np.random.randint(h)
        focus_y = np.random.randint(w)
        while image[i, 0, focus_x, focus_y] == 0 and image[i, 1, focus_x, focus_y]==0:
            focus_x = np.random.randint(h)
            focus_y = np.random.randint(w)
        x_start = max(focus_x - input_size // 2, 0)
        y_start = max(focus_y - input_size // 2, 0)
        x_end = min(focus_x + input_size // 2, h)
        y_end = min(focus_y + input_size // 2, w)
        Input_image[i, :, :, :] = ROI_cropped_image[i, :, x_start:x_end, y_start:y_end]
        Ori_image[i, :, :, :] = image[i, :, x_start:x_end, y_start:y_end]
        for x in range(input_size):
            for y in range(input_size):
                distance = np.sqrt((x - input_size // 2)**2 + (y - input_size // 2)**2)
                if distance <= ROI_inner:
                    Input_image[i, :, x, y] = 0
                    Loss_mask[i, :, x, y] = 1
                elif (distance >  ROI_inner) & (distance <= ROI_outer):
                    Input_image[i, :, x, y] = ((distance - ROI_inner)/(ROI_outer - ROI_inner)) * Input_image[i, :, x, y]
                    Loss_mask[i, :, x, y] = 1
    return Input_image, Ori_image, Loss_mask
    # return ROI_cropped_image, image
        
def load_in_batch(data, batch_size, shuffle=False):
    if shuffle:
        data = np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def normalize_fft(fft_tensor):
    """归一化频域信号"""
    return torch.view_as_real(fft_tensor) / torch.sqrt(torch.tensor(fft_tensor.size(-1)*fft_tensor.size(-2)))

def train_epoch(train, ROI_inner, ROI_outer, input_size, batch_size, device, optimizer, model, criterion):
    train_err = 0
    train_batches = 0

    for data in load_in_batch(train, batch_size):
        input, gt, loss_mask = ROI_crop(data, ROI_inner, ROI_outer, input_size)
        input = torch.from_numpy(input).float().to(device)
        input = input.reshape(batch_size, 2, input_size, input_size)
        gt = torch.from_numpy(gt).float().to(device)
        gt = gt.reshape(batch_size, 2, input_size, input_size)
        loss_mask = torch.from_numpy(loss_mask).to(device)
        
        input = Variable(input)
        optimizer.zero_grad()
        model.train()
        # print(input.device)
        # print(model.device)
        # exit()
        output = model(input, loss_mask)
        gtphase = torch.atan2(gt[:, 1, :, :]+epsilon, gt[:, 0, :, :]+epsilon)
        outputphase = torch.atan2(output[:, 1, :, :]+epsilon, output[:, 0, :, :]+epsilon)
        loss_img = criterion(output*loss_mask, gt*loss_mask)
        loss_phase = criterion(outputphase*loss_mask[:, 0, :, :], gtphase*loss_mask[:, 0, :, :])
        loss = loss_img + alpha * loss_phase

        output_complex = (output[:, 0, :, :] + 1j*output[:, 1, :, :]) * loss_mask
        label_complex = (gt[:, 0, :, :] + 1j*gt[:, 1, :, :]) * loss_mask
        pred_fft = normalize_fft(torch.fft.fft2(output_complex))
        target_fft = normalize_fft(torch.fft.fft2(label_complex))

        amp_loss = F.l1_loss(output_complex.abs(), label_complex.abs())
        phs_loss = (torch.angle(output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().mean(0).mean()
        freq_loss = criterion(pred_fft[...,0], target_fft[...,0]) + criterion(pred_fft[...,1], target_fft[...,1])
        loss = amp_loss + 2 * phs_loss

        loss.backward()
        optimizer.step()

        train_err += float(loss.item())
        train_batches += 1

    return train_err / train_batches

def valid_epoch(valid, ROI_inner, ROI_outer, input_size, batch_size, device, model, criterion):
    valid_err = 0
    valid_batches = 0
    
    for data in load_in_batch(valid, batch_size):
        with torch.no_grad():
            input, gt, loss_mask = ROI_crop(data, ROI_inner, ROI_outer, input_size)
            input = torch.from_numpy(input).float().to(device)
            input = input.reshape(batch_size, 2, input_size, input_size)
            gt = torch.from_numpy(gt).float().to(device)
            gt = gt.reshape(batch_size, 2, input_size, input_size)
            loss_mask = torch.from_numpy(loss_mask).to(device)

            input = Variable(input)
            model.eval()
            output = model(input, loss_mask)
            gtphase = torch.atan2(gt[:, 1, :, :]+epsilon, gt[:, 0, :, :]+epsilon)
            outputphase = torch.atan2(output[:, 1, :, :]+epsilon, output[:, 0, :, :]+epsilon)
            err_img = criterion(output*loss_mask, gt*loss_mask)
            err_phase = criterion(outputphase*loss_mask[:, 0, :, :], gtphase*loss_mask[:, 0, :, :])
            # err = err_img + alpha * err_phase
            # err = criterion(output*loss_mask, gt.squeeze()*loss_mask)

            output_complex = (output[:, 0, :, :] + 1j*output[:, 1, :, :]) * loss_mask
            label_complex = (gt[:, 0, :, :] + 1j*gt[:, 1, :, :]) * loss_mask

            amp_err = F.l1_loss(output_complex.abs(), label_complex.abs())
            phs_err = (torch.angle(output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().mean(0).mean()
        
            pred_fft = normalize_fft(torch.fft.fft2(output_complex))
            target_fft = normalize_fft(torch.fft.fft2(label_complex))
            err_freq = criterion(pred_fft[...,0], target_fft[...,0]) + criterion(pred_fft[...,1], target_fft[...,1])
            err = err_img + alpha * err_phase
            err = amp_err + 2 * phs_err

            valid_err += err.item()
            valid_batches += 1
    return valid_err / valid_batches

def test_epoch(test, ROI_inner, ROI_outer, input_size, batch_size, device, model, criterion):
    test_err = 0
    test_batches = 0
    PhaseImages = []
    InputImages = []
    GTImages = []
    
    for data in load_in_batch(test, batch_size):
        with torch.no_grad():
            input, gt, loss_mask = ROI_crop(data, ROI_inner, ROI_outer, input_size)
            input = torch.from_numpy(input).float().to(device)
            input = input.reshape(batch_size, 2, input_size, input_size)
            gt = torch.from_numpy(gt).float().to(device)
            gt = gt.reshape(batch_size, 2, input_size, input_size)
            loss_mask = torch.from_numpy(loss_mask).to(device)

            input = Variable(input)
            model.eval()
            output = model(input, loss_mask)
            gtphase = torch.atan2(gt[:, 1, :, :]+epsilon, gt[:, 0, :, :]+epsilon)
            outputphase = torch.atan2(output[:, 1, :, :]+epsilon, output[:, 0, :, :]+epsilon)
            err_img = criterion(output*loss_mask, gt*loss_mask)
            err_phase = criterion(outputphase*loss_mask[:, 0, :, :], gtphase*loss_mask[:, 0, :, :])

            output_complex = (output[:, 0, :, :] + 1j*output[:, 1, :, :]) * loss_mask
            label_complex = (gt[:, 0, :, :] + 1j*gt[:, 1, :, :]) * loss_mask
            pred_fft = torch.fft.fft2(output_complex)
            target_fft = torch.fft.fft2(label_complex)
            err_freq = criterion(pred_fft.real.float(), target_fft.real.float()) + criterion(pred_fft.imag.float(), target_fft.imag.float())
            # err = err_img + alpha * err_phase + err_freq
            err = criterion(output*loss_mask, gt.squeeze()*loss_mask)
            for i in range(batch_size):
                PhaseImages.append(output[i, :, :, :])
                InputImages.append(input.squeeze()[i, :, :, :])
                GTImages.append(gt.squeeze()[i, :, :, :])
            
            test_err += err.item()
            test_batches += 1
            test_err /= test_batches
    return test_err, PhaseImages, InputImages, GTImages
