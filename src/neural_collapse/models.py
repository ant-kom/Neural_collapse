import torch.nn as nn
import torchvision.models as models
import torch
from .resnet import ResNet50


def make_model(model_name: str, num_classes: int, one_channel: bool) -> nn.Module:
    model_dict = {
        # resnet
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT, set_resnet),
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT, set_resnet),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT, set_resnet),
        'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT, set_resnet),
        'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT, set_resnet),

        # vgg
        'vgg11': (models.vgg11, models.VGG11_Weights.DEFAULT, set_vgg),
        'vgg11_bn': (models.vgg11_bn, models.VGG11_BN_Weights.DEFAULT, set_vgg),
        'vgg13': (models.vgg13, models.VGG13_Weights.DEFAULT, set_vgg),
        'vgg13_bn': (models.vgg13_bn, models.VGG13_BN_Weights.DEFAULT, set_vgg),
        'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT, set_vgg),
        'vgg16_bn': (models.vgg16_bn, models.VGG16_BN_Weights.DEFAULT, set_vgg),
        'vgg19': (models.vgg19, models.VGG19_Weights.DEFAULT, set_vgg),
        'vgg19_bn': (models.vgg19_bn, models.VGG19_BN_Weights.DEFAULT, set_vgg),
        
        # mobilenet
        'mobilenet_v3_large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, set_mobilenet),
        'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, set_mobilenet),

        #resnet pretrained
        'resnet50_pretrained': (ResNet50, None, set_resnet_pretrained)
    }

    model_class, weights, set_function = model_dict[model_name]
    if weights:
        model = model_class(weights)
    else:
        model = model_class()
    set_function(model, num_classes, one_channel)
    return model

def set_resnet(model: nn.Module, num_classes: int, one_channel: bool):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if one_channel:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

def set_vgg(model: nn.Module, num_classes: int, one_channel: bool):
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    if one_channel:
        old_conv = model.features[0]
        model.features[0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

def set_mobilenet(model: nn.Module, num_classes: int, one_channel: bool):
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    if one_channel:
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

def set_resnet_pretrained(model: nn.Module, num_classes: int, one_channel: bool):
    in_features = model.linear.in_features
    model.linear = nn.Linear(in_features, num_classes)
    if one_channel:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

class ModelWrapper(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 10, one_channel: bool = False, weights_path: str | None = None):
        super().__init__()
        self.model = make_model(model_name, num_classes, one_channel)

        if model_name != 'resnet50_pretrained' and weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict)
        elif weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict['net'])


    def forward(self, x):
        return self.model(x)
