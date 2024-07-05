from torch import nn


# based on https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1 (visited on May 22, 2022)
class VGG16(nn.Module):
    def __init__(self, input_size: int = 9, num_classes: int = 4, **kwargs):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_size, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(384, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 * 7 * 7, 6144),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(6144, 6144),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(6144, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
