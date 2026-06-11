import torch
from torch import nn
from torch.utils.data import DataLoader
from .utils import print_step_info
from tqdm import tqdm
import gc


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
        N = self.class_counts.sum().item()
        self.Var /= (N - 1)
        self.Z /= self.class_counts
        ex = (self.Z * self.class_counts).sum(dim=1, keepdim=True) / N
        self.Var.addmm_(
            ex,
            ex.T,
            alpha=-N /(N - 1),
        )
        del ex

@print_step_info("CALCULATE Z and Var")
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
    ood_dataloader: DataLoader | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()

    # Find number of features
    features = []
    def hook(module, input, output):
        features.append(output.detach().clone())

    target_name = layer_name[1]
    layer: nn.Module = dict([*model.named_modules()])[target_name]
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        num_features = None

        for _, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            features.clear()
            _ = model(x)
            feat = features[layer_name[2]-1]
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
                feat = features[layer_name[2]-1]
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
    new_num_features = num_features
    if add_affine:
        new_num_features = int(num_features * affine_shape_coeff)
        Matrix_A = torch.randn((new_num_features, num_features), device=device)
        Vector_b = torch.randn((new_num_features,), device=device)
    st = CalculateStat(new_num_features, num_classes, device)
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            features.clear()
            _ = model(x)
            feat = features[layer_name[2]-1]
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

    #Calculate ood metric
    ood_metric = 0
    num_ood_objects = 0
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        for _, batch in enumerate(ood_dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            features.clear()
            outputs = model(x)
            feat = features[layer_name[2]-1]
            feat = feat.view(feat.size(0), -1)

            if add_linear:
                logits = linear_class(feat)
            else:
                logits = feat

            k = outputs.argmax(dim=1)
            ood_metric += ((st.Z[:, k].T - logits)**2).sum(dim=1).sum()
            num_ood_objects+= x.shape[0]
        ood_metric = torch.sqrt(ood_metric).item() / num_ood_objects

    handle.remove()


    if verbose:
        print(f"Var shape = {st.Var.shape}")
        print(f"Z shape = {st.Z.shape}")
        print(f"ood_metric = {ood_metric}")

    return st.Z, st.Var, ood_metric

@print_step_info("CALCULATE A0")
def calculate_a0(Z: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    n_features, num_classes = Z.shape
    MAX_NUMBER_OF_FEATURES = 20_000
    if n_features > MAX_NUMBER_OF_FEATURES:
        if verbose:
            print(f"Too many features = {n_features}. Algorithm works with number of features < {MAX_NUMBER_OF_FEATURES}")
        return None
    
    A0 = torch.zeros((n_features, n_features), device=Z.device)
    ZTZ = Z.T @ Z
    try:
        M = torch.linalg.solve(ZTZ, Z.T)
    except:
        if verbose:
            print("Matrix Z.T @ Z is not full rank matrix!!!")
        return None

    A0[:num_classes, :] = M
    B = Z @ M
    A0 -= B
    A0 += torch.eye(n_features, device=Z.device)
    if verbose:
        check_matr = torch.zeros((n_features, Z.shape[1]), device=Z.device)
        check_matr[:Z.shape[1], :] = torch.eye(Z.shape[1])
        A0_nrm = torch.norm(A0)
        A0Z_min_check_nrm = torch.norm(A0 @ Z - check_matr)
        print(f"A0 shape = {A0.shape}")
        print(f"||A0 @ Z - I|| = {A0Z_min_check_nrm}")
        print(f"||A0|| = {A0_nrm}")
        print(f"||A0 @ Z - I|| / ||A0|| = {A0Z_min_check_nrm / A0_nrm}")

    return A0

@print_step_info("Calculate S")
def calculate_s(A0: torch.Tensor, Var: torch.Tensor, num_classes: int, verbose: bool = False) -> torch.Tensor:
    Var = A0 @ Var
    Sigma = Var @ A0.T

    K = num_classes
    Sigma_11 = Sigma[:K, :K]
    Sigma_12 = Sigma[:K, K:]
    Sigma_21 = Sigma[K:, :K]
    Sigma_22 = Sigma[K:, K:]

    U, S, _ = torch.linalg.svd(Sigma_22, full_matrices=False)

    s_max = S[0] if S.numel() > 0 else torch.tensor(0.0, device=Sigma.device)
    rank = torch.sum(S > 1e-10 * s_max).item()

    if rank == S.shape[0]:
        Sigma22_inv = torch.linalg.inv(Sigma_22)
        S = Sigma_11 - Sigma_12 @ Sigma22_inv @ Sigma_21
    else:
        U1 = U[:, :rank]
        Sigma22_proj = U1.T @ Sigma_22 @ U1
        Sigma12_proj = Sigma_12 @ U1
        Sigma22_proj_inv = torch.linalg.inv(Sigma22_proj)
        S = Sigma_11 - Sigma12_proj @ Sigma22_proj_inv @ Sigma12_proj.T

    if verbose:
        print(f"Sigma_11 shape = {Sigma_11.shape}")
        print(f"Sigma_12 shape = {Sigma_12.shape}")
        print(f"Sigma_21 shape = {Sigma_21.shape}")
        print(f"Sigma_22 shape = {Sigma_22.shape}")
        print(f"Sigma_22 rank = {rank}")
        print(f"S shape = {S.shape}")
        print(f"||Sigma11 - Sigma11.T|| / ||Sigma11|| = {torch.norm(Sigma_11 - Sigma_11.T) / torch.norm(Sigma_11)}")
        print(f"||Sigma12 - Sigma21|| / ||Sigma12|| = {torch.norm(Sigma_12 - Sigma_21.T) / torch.norm(Sigma_12)}")
        print(f"||Sigma22 - Sigma22.T|| / ||Sigma22|| = {torch.norm(Sigma_22 - Sigma_22.T) / torch.norm(Sigma_22)}")
        print(f"||S - S.T|| / ||S|| = {torch.norm(S - S.T) / torch.norm(S)}")

    return S

@print_step_info("Calculate metric")
def calculate_final_metr(S: torch.Tensor, verbose: bool = False) -> float:
    num_classes = S.shape[0]
    e = torch.ones((num_classes, 1), device=S.device)

    trace_S = torch.trace(S)
    Se = S @ e
    eSe = (e.T @ Se).squeeze()     
    Se2 = (Se.T @ Se).squeeze()        

    quad = Se2 / eSe

    metric = trace_S - quad
    coeff = (num_classes - 1) / num_classes

    if verbose:
        print(f"Metric without coeff = {metric.item()}")
        print(f"Coeff delta = {coeff}")

    return metric.item() - coeff

@print_step_info("FINAL RESULTS")
def print_final_res(Z: torch.Tensor, metric: float, ood_metric: float, verbose: bool = False) -> None:
    if verbose:
        print(f"metric = {metric}")
        print(f"ood_metric = {ood_metric}")
        print(f"n_features = {Z.shape[0]}")
        print(f"num_classes = {Z.shape[1]}")

@print_step_info("Start Calculate layer")
def compute_metric(
    model: nn.Module, 
    num_classes: int, 
    layer_name: str, 
    dataloader: DataLoader, 
    verbose: bool = False, 
    add_linear: bool = False, 
    add_affine: bool = False,
    ood_dataloader: DataLoader | None = None,
) -> float:
    if verbose:
        print(f"Layer type = {layer_name[1]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    model.eval()

    with torch.no_grad():
        Z, Var, ood_metric = calculate_stat(
            model=model,
            num_classes=num_classes,
            layer_name=layer_name,
            dataloader=dataloader,
            device=device,
            verbose=verbose,
            add_linear=add_linear,
            add_affine=add_affine,
            ood_dataloader=ood_dataloader,
        )

        A0 = calculate_a0(
            Z=Z,
            verbose=verbose,
        )

        metric = torch.nan
        if A0 is not None:
            S = calculate_s(
                A0=A0,
                Var=Var,
                num_classes=Z.shape[1],
                verbose=verbose,
            )

            metric  = calculate_final_metr(
                S=S,
                verbose=verbose,
            )

        print_final_res(Z=Z, metric=metric, ood_metric=ood_metric, verbose=verbose)
        del Z, Var
        return metric

def compute_layers_metrics(
    layers_names: str, 
    model: nn.Module, 
    num_classes: int, 
    dataloader: DataLoader, 
    verbose: bool, 
    add_linear: bool = False, 
    add_affine: bool = False,
    ood_dataloader: DataLoader | None = None,
) -> torch.Tensor:
    values = torch.zeros((1, len(layers_names)))
    for idx, layer in enumerate(tqdm(layers_names)):
        metric = compute_metric(
            model=model, 
            num_classes=num_classes,
            layer_name=layer, 
            dataloader=dataloader, 
            verbose=verbose,
            add_linear=add_linear, 
            add_affine=add_affine,
            ood_dataloader=ood_dataloader,
        ) 
        torch.cuda.empty_cache() 
        gc.collect() 
        values[:, idx] = metric

    return values
