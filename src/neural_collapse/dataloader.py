import torchvision
from torch.utils.data import DataLoader, Dataset
from .generate_rand import get_random_dataset
from torchvision import transforms


class OODWrapper(Dataset):
    def __init__(self, dataset, ood_label=-1):
        self.dataset = dataset
        self.ood_label = ood_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x, self.ood_label


def get_dataloaders(dataset_name: str, batch_size: int = 16, add_ood: str | None = None):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        data_train = torchvision.datasets.CIFAR10('../data', train=True, transform=transform, download=True)
        data_test = torchvision.datasets.CIFAR10('../data', train=False, transform=transform, download=False)
        num_classes = 10
        one_channel = False
    elif dataset_name == "cifar10_augm":
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
    else:
        raise ValueError("Wrong dataset name = {dataset_name}. Should be cifar10, cifar10_augm or random.")
    
    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    
    ood_dataloader_train, ood_dataloader_test = None, None
    if add_ood:
        ood_data_train, ood_data_test = add_ood_examples(add_ood)
        ood_dataloader_train = DataLoader(ood_data_train, batch_size=batch_size, shuffle=True)
        ood_dataloader_test = DataLoader(ood_data_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test, ood_dataloader_train, ood_dataloader_test, num_classes, one_channel


def add_ood_examples(add_ood: str):
    if add_ood == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
        ])
        ood_data_train = OODWrapper(torchvision.datasets.MNIST('../data', train=True, transform=transform, download=True))
        ood_data_test = OODWrapper(torchvision.datasets.MNIST('../data', train=False, transform=transform, download=False))
    elif add_ood == "fashion_mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
        ])
        ood_data_train = OODWrapper(torchvision.datasets.FashionMNIST('../data', train=True, transform=transform, download=True))
        ood_data_test = OODWrapper(torchvision.datasets.FashionMNIST('../data', train=False, transform=transform, download=False))
    elif add_ood == "imagenette":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        ood_data_train = OODWrapper(torchvision.datasets.Imagenette('../data', split="train", transform=transform, download=True))
        ood_data_test = OODWrapper(torchvision.datasets.Imagenette('../data', split="val", transform=transform, download=False))
    else:
        raise ValueError("Wrong ood dataset name = {add_ood}. Should be mnist, fashion_mnist or imagenette.")

    return ood_data_train, ood_data_test
