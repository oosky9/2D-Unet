import torch
from torch import nn

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.enc1 = self.conv_bn_relu(  1,  32, kernel_size=5)  # 32x224x224
        self.enc2 = self.conv_bn_relu( 32,  64, kernel_size=3, pool_kernel=2)  # 64x112x112
        self.enc3 = self.conv_bn_relu( 64, 128, kernel_size=3, pool_kernel=2)  # 128x56x56
        self.enc4 = self.conv_bn_relu(128, 256, kernel_size=3, pool_kernel=2)  # 256x28x28
        self.enc5 = self.conv_bn_relu(256, 512, kernel_size=3, pool_kernel=2)  # 512x14x14

        self.dec1 = self.conv_bn_relu(512, 256, kernel_size=3, pool_kernel=-2)  # 256x28x28
        self.dec2 = self.conv_bn_relu(256 + 256, 128, kernel_size=3, pool_kernel=-2)  # 128x56x56
        self.dec3 = self.conv_bn_relu(128 + 128, 64, kernel_size=3, pool_kernel=-2)  # 64x112x112
        self.dec4 = self.conv_bn_relu( 64 +  64, 32, kernel_size=3, pool_kernel=-2)  # 32x224x224
        self.dec5 = nn.Sequential(
            nn.Conv2d(32 + 32, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None):
        layers = []
        if pool_kernel is not None:
            if pool_kernel > 0:
                layers.append(nn.AvgPool2d(pool_kernel))
            elif pool_kernel < 0:
                layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size-1)//2))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)


    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        out = self.dec1(x5)
        out = self.dec2(torch.cat([out, x4], dim=1))
        out = self.dec3(torch.cat([out, x3], dim=1))
        out = self.dec4(torch.cat([out, x2], dim=1))
        out = self.dec5(torch.cat([out, x1], dim=1))
        return out
