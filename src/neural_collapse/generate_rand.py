import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from math import sqrt


def get_random_dataset(n_classes: int, n_features: int, n_samples_per_class: int):
    test_ratio = 0.2
    seed = 42

    torch.manual_seed(seed)

    X_list = []
    y_list = []

    for i in range(n_classes):
        mean = torch.randn(n_features)
        cov = torch.eye(n_features)

        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

        samples = dist.sample((n_samples_per_class,)).reshape((n_samples_per_class, int(sqrt(n_features)), int(sqrt(n_features))))
        labels = torch.full((n_samples_per_class,), i, dtype=torch.long)

        X_list.append(samples)
        y_list.append(labels)

    X = torch.cat(X_list, dim=0).unsqueeze(1)
    y = torch.cat(y_list, dim=0)

    dataset = TensorDataset(X, y)

    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, test_dataset
