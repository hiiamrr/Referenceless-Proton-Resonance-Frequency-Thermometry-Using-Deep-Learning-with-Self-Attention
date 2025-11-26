import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from torch.nn import functional as F
from model.unet import UNet
from model.resunet import RESUNet
from model.swtnet import SwinIR

class Refless_Model(nn.Module):
    def __init__(self):
        super(Refless_Model, self).__init__()
        self.net = RESUNet()

    def forward(self, x, mask):
        y = self.net(x)
        y = y.squeeze()
        y = y*mask + x.squeeze()*(1-mask)
        return y
