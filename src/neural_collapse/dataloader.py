import torchvision
from torch.utils.data import DataLoader
from .generate_rand import get_random_dataset
from torchvision import transforms


def get_dataloaders(dataset_name: str, batch_size: int = 16):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        data_train = torchvision.datasets.CIFAR10('../data', train=True, transform=transform, download=True)
        data_test = torchvision.datasets.CIFAR10('../data', train=False, transform=transform, download=False)
        num_classes = 10
        one_channel = False
    if dataset_name == "cifar10_augm":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_train = torchvision.datasets.CIFAR10('../data', train=True, transform=transform, download=True)
        data_test = torchvision.datasets.CIFAR10('../data', train=False, transform=transform, download=False)
        num_classes = 10
        one_channel = False
    elif dataset_name == "random":
        num_classes = 10
        one_channel = True
        data_train, data_test = get_random_dataset(num_classes, 1024, 1000)
    elif dataset_name == "imagenet":
        one_channel = False
        num_classes = 1000
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        pass
    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test, num_classes, one_channel
