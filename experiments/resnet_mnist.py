import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from neural_collapse.metric import compute_metric
from neural_collapse.models import ModelWrapper
from neural_collapse.trainer import Trainer
from neural_collapse.plots import make_plots
from tqdm import tqdm
import gc


device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#data_train = torchvision.datasets.MNIST('../data', train=True, transform=transform, download=True)
#data_test = torchvision.datasets.MNIST('../data', train=False, transform=transform, download=False)
data_train = torchvision.datasets.CIFAR10('../data', train=True, transform=transform, download=True)
data_test = torchvision.datasets.CIFAR10('../data', train=False, transform=transform, download=False)
NUM_CLASSES = 10

dataloader_train = DataLoader(data_train, batch_size=16, shuffle=True)
dataloader_test = DataLoader(data_test, batch_size=16, shuffle=True)

model = ModelWrapper(NUM_CLASSES, False, "artifacts/cifar10_resnet18.pth")


trainer = Trainer(model, dataloader_train, None, dataloader_test)

#trainer.train_function(num_epochs=5, save_model_weigths="artifacts/cifar10_resnet18.pth", verbose=True)
#trainer.eval_function(verbose=True)
layers_names = [
    name for name, module in trainer.model.model.named_modules()
    if len(list(module.children())) == 0
]

values = torch.zeros((1, len(layers_names)))
for idx, layer in enumerate(tqdm(layers_names)):
    res = compute_metric(
        model=trainer.model.model, 
        num_classes=NUM_CLASSES,
        layer_name=layer, 
        dataloader=dataloader_test, 
        calc_final=True, 
        verbose=False
    ) 
    torch.cuda.empty_cache() 
    gc.collect() 
    values[:, idx] = res.metric
    del res

make_plots(
    values=values.tolist(),
    layers=layers_names,
    header="Resnet18 on Cifar10",
    values_name="Metric",
    save_path="metric_all.pdf",
    classes_names=None,
    selected_layers=['layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1', 'fc'],
)
