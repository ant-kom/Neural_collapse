import torch
from torch import nn
from torch.utils.data import DataLoader
from .utils import print_step_info
from dataclasses import dataclass
import math
from tqdm import tqdm


@dataclass
class Statistics:
    RMSK: torch.Tensor
    num_classes: int
    n_features: int
    metric: int | None = None

def calculate_stat(model: nn.Module, num_classes: int, layer_name: str, dataloader: DataLoader, device: str, verbose: bool = False) -> Statistics:
    #print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB") 
    #print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    with torch.no_grad():
        # Hook for features
        features = []

        def hook(module, input, output):
            features.append(output.detach())

        layer: nn.Module = dict([*model.named_modules()])[layer_name]
        handle = layer.register_forward_hook(hook)

        class_counts = torch.zeros(num_classes, device=device)
        class_norm = torch.zeros(num_classes, device=device)

        n_features = None

        for idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            features.clear()
            _ = model(x)
            feat = features[0]

            if verbose and idx == 0:
                print(f"Feature shape before = {feat.shape}")

            feat = feat.view(feat.size(0), -1)  # [B, n]
            B, n = feat.shape

            if verbose and idx == 0:
                print(f"Feature shape after reshape = {feat.shape}")

            if n_features is None:
                n_features = n
                print(f"Num features: {n_features}")
                Z = torch.zeros((n_features, num_classes), device=device)
                Var = torch.zeros((n_features, n_features), device=device)

            for k in range(num_classes):
                mask = (y == k)
                if mask.any():
                    selected: torch.Tensor = feat[mask]
                    Z[:, k] += selected.sum(dim=0)
                    class_norm[k] += selected.norm() ** 2
                    class_counts[k] += selected.size(0)
                    del selected

            Var.addmm_(
                feat.T,
                feat,
            )

            del feat


        handle.remove()
        if verbose:
            print(f"Class counts = {class_counts}")

        Var /= class_counts.sum().item()
        ex = Z.sum(dim=1, keepdim=True) / class_counts.sum().item()
        Var.addmm_(
            ex,
            ex.T,
            alpha=-1.0,
        )
        del ex
        if verbose:
            print(f"Var shape = {Var.shape}")

        Z /= class_counts
        if verbose:
            print(f"Z shape = {Z.shape}")

        RMSK = torch.zeros((num_classes), device=device)
        for k in range(num_classes):
            RMSK[k] = torch.sqrt((class_norm[k] / class_counts[k] - torch.norm(Z[:, k]) ** 2) / n_features)

        return Statistics(
            RMSK=RMSK.to("cpu"), 
            num_classes=num_classes,
            n_features=n_features,
        ), Z, Var

def calculate_a0(Z: torch.Tensor, n_features: int, num_classes: int, device: str, verbose: bool = False):
    A0 = torch.zeros((n_features, n_features), device=device)
    ZtZ = Z.T @ Z  # K x K
    A0[:num_classes, :] = torch.linalg.solve(ZtZ, Z.T)
    A0 -= Z @ A0[:num_classes, :]
    A0.diagonal().add_(1.0)
    if verbose:
        print(f"A0 shape = {A0.shape}")

    return A0

def calculate_s(A0: torch.Tensor, Var: torch.Tensor, num_classes: int, verbose: bool = False):
    Var = A0 @ Var
    Sigma = Var @ A0.T

    K = num_classes
    Sigma_11 = Sigma[:K, :K]
    Sigma_12 = Sigma[:K, K:]
    Sigma_21 = Sigma[K:, :K]
    Sigma_22 = Sigma[K:, K:]

    if verbose:
        print(f"Sigma_11 shape = {Sigma_11.shape}")
        print(f"Sigma_12 shape = {Sigma_12.shape}")
        print(f"Sigma_21 shape = {Sigma_21.shape}")
        print(f"Sigma_22 shape = {Sigma_22.shape}")
        print(torch.norm(Sigma_12 - Sigma_21.T) / torch.norm(Sigma_12))
        print(Sigma_22.numel(), (Sigma_22 > 1e-8).sum().item())
        print(Sigma_11.numel(), (Sigma_11 > 1e-8).sum().item())
        print(Sigma_12.numel(), (Sigma_12 > 1e-8).sum().item())
        print(sum(torch.svd(Sigma_22, compute_uv=False).S > 1e-2))


    S = Sigma_11
    if Sigma_22.numel() > 0:
        Sigma_22_inv_Sigma_21 = torch.linalg.pinv(Sigma_22) @ Sigma_21
        S -= Sigma_12 @ Sigma_22_inv_Sigma_21
    if verbose:
        print(f"S shape = {S.shape}")

    return S

def calculate_final_metr(S: torch.Tensor, num_classes: int, device: str, verbose: bool = False):
    ones = torch.ones((num_classes, 1), device=device)

    trace_S = torch.trace(S)
    quad = (ones.T @ S @ ones) / num_classes
    metric = trace_S - quad.squeeze()

    if verbose:
        print(f"Final metric = {metric.item()}")
    return metric.item()

@print_step_info("FINAL RESULTS")
def print_final_res(statistics: Statistics, calc_final: bool = True) -> None:
    print(f"RMSK = {statistics.RMSK}")
    if calc_final:
        print(f"metric = {statistics.metric}")
    print(f"num_classes = {statistics.num_classes}")
    print(f"n_features = {statistics.n_features}")

@print_step_info("CALCULATE METRIC")
def compute_metric(model: nn.Module, num_classes: int, layer_name: str, dataloader: DataLoader, calc_final: bool = True, verbose: bool = False) -> Statistics:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        statistics, Z, Var = calculate_stat(
            model=model,
            num_classes=num_classes,
            layer_name=layer_name,
            dataloader=dataloader,
            device=device,
            verbose=verbose,
        )

        if calc_final:
            A0 = calculate_a0(
                Z=Z,
                n_features=statistics.n_features,
                num_classes=statistics.num_classes,
                device=device,
                verbose=verbose,
            )

            S = calculate_s(
                A0=A0,
                Var=Var,
                num_classes=statistics.num_classes,
                verbose=verbose,
            )

            statistics.metric  = calculate_final_metr(
                S=S,
                num_classes=statistics.num_classes,
                device=device,
                verbose=verbose,
            )

        del Z, Var

        if verbose:
            print_final_res(statistics=statistics, calc_final=calc_final)

    return statistics
