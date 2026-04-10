import torch.nn as nn
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from .utils import print_step_info


class Trainer:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, validation_dataloader: DataLoader, test_dataloader: DataLoader):
        self._model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    @print_step_info("TRAIN MODEL")
    def train_function(
        self,
        lr: float = 0.001,
        momentum: float = 0.9,
        num_epochs: int = 1,
        save_model_weigths: str = "artifacts/weights.pth",
        verbose: bool = False,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        metrics_dict = defaultdict(list)

        for _ in range(num_epochs):

            if verbose:
                print(f"EPOCH {_ + 1}:")

            self.model.train()

            running_loss = 0.0
            total_train = 0.0
            self._reset_metrics()

            # train
            for batch in tqdm(self.train_dataloader, desc="Train", disable=not verbose):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss: nn.Module = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_train += inputs.data.size(0)
                self._update_metrics(outputs, labels)

            self._compute_metrics()
            metrics_dict["Train Accuracy"].append(self._compute_metrics()["Accuracy_user"])
            metrics_dict["Train Loss"].append(float(running_loss / total_train))
            
            if verbose:
                print(f"Train accuracy = {metrics_dict['Train Accuracy'][-1]}")
                print(f"Train loss = {metrics_dict['Train Loss'][-1]}")

            # validation
            if self.validation_dataloader:
                val_dict = self._eval_function(self.validation_dataloader, "Valid", verbose)
                if verbose:
                    print(f"Validation accuracy = {val_dict['Accuracy_user']}")
                metrics_dict["Validation Accuracy"].append(val_dict["Accuracy_user"])

        weights_dir = Path('.')
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / save_model_weigths
        torch.save(self.model.state_dict(), weights_path)

        return metrics_dict
    
    @print_step_info("EVAL MODEL")
    def eval_function(self, verbose: bool = False):
        metrics =  self._eval_function(self.test_dataloader, "Test", verbose)
        if verbose:
            print(f"Test accuracy = {metrics['Accuracy_user']}")
        return metrics
    
    @property
    def model(self):
        return self._model
    
    def _eval_function(self, dataloader: DataLoader, info:str, verbose: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        self._reset_metrics()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=info, disable=not verbose):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                self._update_metrics(outputs, labels)
        
        return self._compute_metrics()

    def _reset_metrics(self) -> None:
        self.total = 0
        self.correct = 0

    def _update_metrics(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        predicted = predicted.argmax(1)
        targets = targets.to(predicted.device)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()

    def _compute_metrics(self, name: str = "Accuracy_user") -> dict[str, float]:
        return {name: float(self.correct / self.total * 100.0)}
