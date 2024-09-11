import torch.nn as nn
import torch
from torchvision import transforms
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512], use_softmax=False):
        super(Unet, self).__init__()

        self.use_softmax = use_softmax
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            # print(x.shape)
            # print(skip_connection.shape)
            if x.shape != skip_connection.shape:
                x = transforms.Resize(
                    size=skip_connection.shape[2:], antialias=False)(x)
            concat_skip = torch.cat((x, skip_connection), dim=1)
            x = self.ups[i+1](concat_skip)

        if self.use_softmax:
            x = self.final_conv(x)
            return F.softmax(x, dim=1)
        else:
            return self.final_conv(x)


if __name__ == '__main__':
    x = torch.randn((3, 1, 161, 161))
    out_channels = 5
    model = Unet(in_channels=1, out_channels=5)
    preds = model(x)
    print(preds.shape)
    assert ((x.shape[0], out_channels, x.shape[2], x.shape[3]) == preds.shape)
