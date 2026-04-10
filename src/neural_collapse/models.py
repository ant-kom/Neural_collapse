import torch.nn as nn
import torchvision.models as models
import torch
from .resnet import ResNet50


class ModelWrapper(nn.Module):
    def __init__(self, num_classes: int = 10, one_channel: bool = False, weights_path: str | None = None):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if one_channel:
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        #self.model = ResNet50()

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
