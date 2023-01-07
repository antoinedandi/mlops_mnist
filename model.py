import torch
import torch.nn as nn


def conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv(in_c, out_c)
        self.res = conv(out_c, out_c)
        self.down = nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return self.res(x) + x


class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            ResBlock(1, 4),
            ResBlock(4, 8),
            ResBlock(8, 16),
            nn.Flatten(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = Classifier()
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    print(y.shape)