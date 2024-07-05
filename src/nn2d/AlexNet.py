# https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html#AlexNet_Weights
import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(
        self, input_size: int = 9, num_classes: int = 4, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 288, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(288, 576, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(576, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(384 * 6 * 6, 6144),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(6144, 6144),
            nn.ReLU(inplace=True),
            nn.Linear(6144, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
