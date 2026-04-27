import torch
from torch import nn
from torch.utils.data import DataLoader
from .utils import print_step_info
from dataclasses import dataclass
from tqdm import tqdm
import gc


@dataclass
class Statistics:
    num_classes: int
    n_features: int
    metric: int | None = None

class CalculateStat:
    def __init__(self, num_features: int, num_classes: int, device: str):
        self.num_classes = num_classes
        self.Z = torch.zeros((num_features, num_classes), device=device)
        self.Var = torch.zeros((num_features, num_features), device=device)
        self.class_counts = torch.zeros(num_classes, device=device)

    def update_stat(self, x: torch.Tensor, y: torch.Tensor):
        x = x.view(x.size(0), -1) 
        for k in range(self.num_classes):
            mask = (y == k)
            if mask.any():
                selected: torch.Tensor = x[mask]
                self.Z[:, k] += selected.sum(dim=0)
                self.class_counts[k] += selected.size(0)
                del selected

        self.Var.addmm_(
            x.T,
            x,
        )
        del x

    def collect_stat(self):
        self.Var /= self.class_counts.sum().item()
        self.Z /= self.class_counts
        ex = (self.Z * self.class_counts).sum(dim=1, keepdim=True) / self.class_counts.sum()
        self.Var.addmm_(
            ex,
            ex.T,
            alpha=-1.0,
        )
        del ex

def calculate_stat(
    model: nn.Module, 
    num_classes: int, 
    layer_name: str, 
    dataloader: DataLoader, 
    device: str, 
    verbose: bool = False, 
    add_linear: bool = False,
    add_affine: bool = False,
    affine_shape_coeff: int = 2,
) -> Statistics:

    # Find number of features
    features = []
    def hook(module, input, output):
        features.append(output.detach())

    layer: nn.Module = dict([*model.named_modules()])[layer_name]
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        num_features = None

        for _, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            features.clear()
            _ = model(x)
            feat = features[0]
            feat = feat.view(feat.size(0), -1) 
            num_features = feat.shape[1]
            break
    handle.remove()

    # Make linear module
    if add_linear:
        linear_class = nn.Linear(num_features, num_classes).to(device)
        linear_class.train()
        linear_class.requires_grad_(True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(linear_class.parameters(), lr=1e-2)
        total_loss = 0.0
        handle = layer.register_forward_hook(hook)
        for _, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                features.clear()
                _ = model(x)
                feat = features[0]
                feat = feat.view(feat.size(0), -1)

            with torch.enable_grad():
                feat = feat.detach().requires_grad_(True)
                logits = linear_class(feat)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
        handle.remove()
        num_features = num_classes

    # Calculate mean and std
    if add_affine:
        num_features = int(num_features * affine_shape_coeff)
        Matrix_A = torch.randn((num_features, num_classes), device=device)
        Vector_b = torch.randn((num_features,), device=device)
    st = CalculateStat(num_features, num_classes, device)
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            features.clear()
            _ = model(x)
            feat = features[0]
            feat = feat.view(feat.size(0), -1)

            if add_linear:
                logits = linear_class(feat)
            else:
                logits = feat

            if add_affine:
                logits = (Matrix_A @ logits.T + Vector_b.reshape(-1, 1)).T

            st.update_stat(logits, y)
    handle.remove()

    st.collect_stat()

    if verbose:
        print(f"Var shape = {st.Var.shape}")
        print(f"Z shape = {st.Z.shape}")

    return Statistics(
        num_classes=num_classes,
        n_features=num_features,
    ), st.Z, st.Var

def calculate_a0(Z: torch.Tensor, n_features: int, num_classes: int, device: str, verbose: bool = False, eps: float = 1e-4):
    if n_features > 20000:
        return None, 1
    A0 = torch.zeros((n_features, n_features), device=device)
    U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    p = sum(S > eps)
    if p < num_classes:
        return None, p

    Zinv_Z = Vh.T @ torch.diag(1.0 / S) @ U.T
    A0[:num_classes, :] = Zinv_Z
    B = Z @ A0[:num_classes, :]
    A0 -= B
    A0 += torch.eye(n_features, device=device)
    if verbose:
        print(f"A0 shape = {A0.shape}")
        check_matr = torch.zeros((n_features, Z.shape[1]), device=device)
        check_matr[:Z.shape[1], :] = torch.eye(Z.shape[1])
        print(f"CHECKING A0 = ", torch.norm(A0 @ Z - check_matr), torch.norm(A0), torch.norm(A0 @ Z - check_matr) / torch.norm(A0))

    return A0, p

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
        print(sum(torch.svd(Sigma_11, compute_uv=False).S > 1e-6))
        print(sum(torch.svd(Sigma_21, compute_uv=False).S > 1e-6))
        print(sum(torch.svd(Sigma_22, compute_uv=False).S > 1e-6))


    S = Sigma_11
    if Sigma_22.numel() > 0:
        Sigma_22_inv_Sigma_21 = torch.linalg.pinv(Sigma_22) @ Sigma_21
        S -= Sigma_12 @ Sigma_22_inv_Sigma_21
    if verbose:
        print(f"S shape = {S.shape}")

    return S

def calculate_final_metr(S: torch.Tensor, num_classes: int, device: str, verbose: bool = False):
    ones_K = torch.ones((num_classes, 1), device=device)

    trace_S = torch.trace(S)
    quad =  torch.norm(S @ ones_K)**2 / (ones_K.T @ S @ ones_K)
    metric = trace_S - quad.squeeze()
    coeff = (num_classes - 1)/num_classes

    if verbose:
        print(f"Final metric = {metric.item()}")
    return metric.item() - coeff

@print_step_info("FINAL RESULTS")
def print_final_res(statistics: Statistics, calc_final: bool = True) -> None:
    if calc_final:
        print(f"metric = {statistics.metric}")
    print(f"num_classes = {statistics.num_classes}")
    print(f"n_features = {statistics.n_features}")

@print_step_info("CALCULATE METRIC")
def compute_metric(
    model: nn.Module, 
    num_classes: int, 
    layer_name: str, 
    dataloader: DataLoader, 
    calc_final: bool = True, 
    verbose: bool = False, 
    add_linear: bool = False, 
    add_affine: bool = False,
) -> Statistics:
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
            add_linear=add_linear,
            add_affine=add_affine,
        )

        if calc_final:
            A0, p = calculate_a0(
                Z=Z,
                n_features=statistics.n_features,
                num_classes=statistics.num_classes,
                device=device,
                verbose=verbose,
            )

            if p < statistics.num_classes:
                statistics.metric = torch.nan
                del Z, Var
                if verbose:
                    print_final_res(statistics=statistics, calc_final=calc_final)
                return statistics

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

def compute_layers_metrics(
    layers_names: str, 
    model: nn.Module, 
    num_classes: int, 
    dataloader: DataLoader, 
    verbose: bool, 
    add_linear: bool = False, 
    add_affine: bool = False,
) -> torch.Tensor:
    values = torch.zeros((1, len(layers_names)))
    for idx, layer in enumerate(tqdm(layers_names)):
        res = compute_metric(
            model=model, 
            num_classes=num_classes,
            layer_name=layer, 
            dataloader=dataloader, 
            calc_final=True, 
            verbose=verbose,
            add_linear=add_linear, 
            add_affine=add_affine,
        ) 
        torch.cuda.empty_cache() 
        gc.collect() 
        values[:, idx] = res.metric
        print(res.metric)
        del res

    return values
