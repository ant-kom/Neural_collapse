import torch
from torch import nn
from torchvision.models import *
from .resnet import ResNet50
import math
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision
import matplotlib.pyplot as plt


def print_step_info(step_name: str):
    def dec(func):
        def wrapper(*args, **kwargs):
            num = 22 + len(step_name)
            verbose = kwargs.get('verbose', False)
            if verbose:
                print('-' * num)
                print('-' * 10, step_name, '-' * 10)
                print('-' * num)
            return func(*args, **kwargs)
        return wrapper
    return dec

def get_layers(model: nn.Module, start_layer: int):
    layers = [
        (name, module.__class__.__name__) for name, module in model.named_modules()
        if len(list(module.children())) == 0
    ][start_layer:]
    return zip(*layers)

def check_parameters():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = {
        'cifar': torch.randn((1, 3, 32, 32)),
        #'imagenet': torch.randn((1, 3, 224, 224)),
    }
    models = {
        'ResNet50': lambda: ResNet50(),
        # AlexNet
        #'AlexNet': lambda: alexnet(AlexNet_Weights.DEFAULT),

        # DenseNet
        #'DenseNet201': lambda: densenet201(DenseNet201_Weights.DEFAULT),

        # GoogLeNet
        #'GoogLeNet': lambda: googlenet(GoogLeNet_Weights.DEFAULT),

        # MNASNet
        #'MNASNet1_3': lambda: mnasnet1_3(MNASNet1_3_Weights.DEFAULT),

        # MobileNet V2
        #'MobileNet_V2': lambda: mobilenet_v2(MobileNet_V2_Weights.DEFAULT),

        # MobileNet V3
        #'MobileNet_V3_Large': lambda: mobilenet_v3_large(MobileNet_V3_Large_Weights.DEFAULT),
        #'MobileNet_V3_Small': lambda: mobilenet_v3_small(MobileNet_V3_Small_Weights.DEFAULT),

        # RegNet
        #'RegNet_X_32GF': lambda: regnet_x_32gf(RegNet_X_32GF_Weights.DEFAULT),

        # ResNet
        #'ResNet18': lambda: resnet18(ResNet18_Weights.DEFAULT),
        ##'ResNet34': lambda: resnet34(ResNet34_Weights.DEFAULT),
        #'ResNet50': lambda: resnet50(ResNet50_Weights.DEFAULT),
        #'ResNet101': lambda: resnet101(ResNet101_Weights.DEFAULT),
        #'ResNet152': lambda: resnet152(ResNet152_Weights.DEFAULT),

        # ResNeXt
        #'ResNeXt101_64x4d': lambda: resnext101_64x4d(ResNeXt101_64X4D_Weights.DEFAULT),


        # SqueezeNet
        #'SqueezeNet1_1': lambda: squeezenet1_1(SqueezeNet1_1_Weights.DEFAULT),

        # VGG
        #'VGG19':lambda:  vgg19(VGG19_Weights.DEFAULT),
        #'VGG19_BN': lambda: vgg19_bn(VGG19_BN_Weights.DEFAULT),

        #'Wide_ResNet101_2': lambda: wide_resnet101_2(Wide_ResNet101_2_Weights.DEFAULT),
    }
    res = {}
    for model_name, model in models.items():
        for image_name, image in images.items():
            model = model()
            model = model.to(device)
            model.eval()
            layers_names = [
                name for name, module in model.named_modules()
                if len(list(module.children())) == 0
            ]
            with torch.no_grad():
                features_n = []
                for layer_name in layers_names:
                    features = []

                    def hook(module, input, output):
                        features.append(output.detach())

                    layer: nn.Module = dict([*model.named_modules()])[layer_name]
                    handle = layer.register_forward_hook(hook)

                    image = image.to(device)

                    features.clear()
                    _ = model(image)
                    feat = features[0]

                    feat = feat.view(feat.size(0), -1)  # [B, n]
                    B, n = feat.shape

                    features_n.append(n)

                    handle.remove()

            if model_name not in res:
                res[model_name] = dict()
            res[model_name] = features_n
            print(f"Model = {model_name},  feature_max = {max(features_n)}")
    return res

def get_forward_trace(model, start_layer: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn((1, 3, 32, 32)).to(device)
    model.eval()

    trace = []
    hooks = []

    global_call_id = 0
    layer_call_counter = {}

    def make_hook(name):
        def hook(module, input, output):
            nonlocal global_call_id

            # локальный счётчик для слоя
            layer_call_counter[name] = layer_call_counter.get(name, 0) + 1
            local_call_id = layer_call_counter[name]

            in_tensor = input[0] if isinstance(input, (tuple, list)) else input
            out_tensor = output[0] if isinstance(output, (tuple, list)) else output

            input_features = math.prod(in_tensor.shape[1:])
            output_features = math.prod(out_tensor.shape[1:])

            trace.append((
                global_call_id,        # глобальный порядок
                name,                  # имя слоя
                module.__class__.__name__,  # тип модуля
                local_call_id,         # номер вызова этого слоя
                output_features >= input_features, # Число признаков увеличилось
            ))

            global_call_id += 1

        return hook

    # регистрируем hook на ВСЕ leaf-модули
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    trace = trace[start_layer:]
    layer_types = [t for _, _, t, _, _ in trace]
    layer_meta = [(g, n, l) for g, n, _, l, _ in trace]
    layer_fetures_change = [t for _, _, _, _, t in trace]

    return layer_meta, layer_types, layer_fetures_change




def _save_batch(images, labels, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        img = images[i]

        plt.figure(figsize=(4, 4))

        if img.shape[0] == 1:
            plt.imshow(
                img.squeeze(0).cpu(),
                cmap="gray"
            )
        else:
            plt.imshow(
                img.permute(1, 2, 0)
                   .cpu()
                   .clamp(0, 1)
            )

        plt.title(f"label={int(labels[i])}")
        plt.axis("off")

        plt.savefig(
            out_dir / f"{i:04d}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


def save_image_examples(
    dataloader,
    ood_dataloader,
    output_dir="image_examples",
):
    output_dir = Path(output_dir)

    x, y = next(iter(dataloader))
    _save_batch(
        x,
        y,
        output_dir / "id"
    )

    x, y = next(iter(ood_dataloader))
    _save_batch(
        x,
        y,
        output_dir / "ood"
    )