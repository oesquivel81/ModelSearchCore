import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.net(x)
        return torch.flatten(x, 1)


class BaselinePatchCNN(nn.Module):
    def __init__(self, base_channels=32, num_classes=13):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=1, base_channels=base_channels)
        self.head = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class VarianceInputPatchCNN(nn.Module):
    def __init__(self, base_channels=32, num_classes=13):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=2, base_channels=base_channels)
        self.head = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class VarianceBranchPatchCNN(nn.Module):
    def __init__(self, base_channels=32, num_classes=13):
        super().__init__()
        self.image_encoder = ConvEncoder(in_channels=1, base_channels=base_channels)
        self.variance_encoder = ConvEncoder(in_channels=1, base_channels=base_channels)

        self.head = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(base_channels * 4, num_classes)
        )

    def forward(self, image, variance):
        zi = self.image_encoder(image)
        zv = self.variance_encoder(variance)
        z = torch.cat([zi, zv], dim=1)
        return self.head(z)
