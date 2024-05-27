from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b3


class EfficientNet(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 1, architecture: str = "b3"):
        super(EfficientNet, self).__init__()
        if architecture == "b0":
            self.base = efficientnet_b0(pretrained=True)
            first_conv = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        elif architecture == "b3":
            self.base = efficientnet_b3(pretrained=True)
            first_conv = nn.Conv2d(input_channels, 40, kernel_size=(3, 3), stride=(2, 2), bias=False)

        self.base.features[0][0] = first_conv
        self.base.classifier = nn.Linear(self.base.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base(x)
