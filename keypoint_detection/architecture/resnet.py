from torch import nn
from torchvision.models import resnet18, resnet50


class ResNet(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 1, architecture: str = "resnet50"):
        super(ResNet, self).__init__()
        if architecture == "resnet18":
            self.base = resnet18(pretrained=True)
            first_conv = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif architecture == "resnet50":
            self.base = resnet50(pretrained=True)
            first_conv = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.base.conv1 = first_conv
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return nn.functional.sigmoid(self.base(x))
