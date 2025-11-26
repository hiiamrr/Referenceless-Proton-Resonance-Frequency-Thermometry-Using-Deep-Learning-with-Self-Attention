import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from torch.nn import functional as F

def conv(batchnorm, input_channels, output_channels, kernel_size=3):
    if batchnorm:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )

def predict_image(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 2, kernel_size=5, padding='same', bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )

class Refless_Model(nn.Module):
    def __init__(self):
        super(Refless_Model, self).__init__()
        self.net = UNet()

    def forward(self, x):
        y = self.net(x)
        y = y.squeeze()
        return y
    
class UNet(nn.Module):
    def __init__(self, batchnorm=True):
        super(UNet, self).__init__()
        self.batchnorm = batchnorm
        self.conv0 = conv(self.batchnorm, 2, 16, kernel_size=7)
        self.conv0_1 = conv(self.batchnorm, 16, 16, kernel_size=5)
        self.conv1 = conv(self.batchnorm, 16, 32, kernel_size=5)
        self.conv1_1 = conv(self.batchnorm, 32, 32, kernel_size=5)
        self.conv2 = conv(self.batchnorm, 32, 64)
        self.conv2_1 = conv(self.batchnorm, 64, 64)
        self.conv3 = conv(self.batchnorm, 64, 128)
        self.conv3_1 = conv(self.batchnorm, 128, 128)
        self.conv4 = conv(self.batchnorm, 128, 256)
        self.conv5 = conv(self.batchnorm, 16, 16, kernel_size=5)
        self.deconv4 = deconv(384, 32)
        self.deconv3 = deconv(96, 32)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(48, 16)
        self.predict_image = predict_image(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 0.1)
                constant_(m.bias, 0)
            
    def forward(self, x):
        out_conv0 = self.conv0_1(self.conv0(x))
        out_conv0 = F.max_pool2d(out_conv0, 2)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv1 = F.max_pool2d(out_conv1, 2)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv2 = F.max_pool2d(out_conv2, 2)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv3 = F.max_pool2d(out_conv3, 2)
        out_conv4 = self.conv4(out_conv3)
        concat4 = torch.cat((out_conv4, out_conv3), 1)
        out_deconv3 = self.deconv4(concat4)
        concat3 = torch.cat((out_deconv3, out_conv2), 1)
        out_deconv2 = self.deconv3(concat3)
        concat2 = torch.cat((out_deconv2, out_conv1), 1)
        out_deconv1 = self.deconv2(concat2)
        concat1 = torch.cat((out_deconv1, out_conv0), 1)
        out_conv = self.conv5(self.deconv1(concat1))
        output_image = self.predict_image(out_conv)
        return output_image
