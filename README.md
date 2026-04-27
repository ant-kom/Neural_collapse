# Neural Collapse Experiments — README

## 1. Overview

This project is designed for experiments on **neural collapse** in convolutional neural networks (e.g., ResNet-18) on standard datasets (e.g., CIFAR-10).

---

## 2. Installation and Build (uv)

The project uses the **uv** package manager.

Install dependencies:

```bash
uv sync
```

## 3. Basic Usage Example

```bash
neural-collapse resnet18 cifar10 --weights cifar10_resnet18.pth --verbose
```

---

## 4. Required Arguments

| Argument     | Description        | Example  |
| ------------ | ------------------ | -------- |
| model_name   | Model architecture | resnet18 |
| dataset_name | Dataset name       | cifar10  |

All avaliable lists of models and datasets you can find in model.py and dataloader.py.

---

## 5. CLI Options

### 5.1 weights

Load pretrained model weights.

```bash
--weights cifar10_resnet34.pth
```

Path is interpreted as a relative path to the `artifacts/` directory.

---

### 5.2 batch_size

Batch size for dataloaders.

```bash
--batch_size 16
```

Default: `16`

---

### 5.3 start_layer

Layer index from which analysis starts.

```bash
--start_layer 0
```

Default: `0`

---

### 5.4 train

Enable training mode.

```bash
--train path/to/save_weights.pth
```

If provided, the model is trained and weights are saved to the given path.

---

### 5.5 verbose

Enable detailed logging.

```bash
--verbose
```

Default: `False`

---

### 5.6 linear

Adds a linear classifier after each layer.

```bash
--linear
```

Used for analysis of linear separability of representations.

---

### 5.7 affine

Tests invariance to linear transformations.

```bash
--affine
```

Used to evaluate stability of representations under linear mappings.

---

## 6. Usage Examples

### 6.1 Basic inference with pretrained weights

```bash
neural-collapse resnet34 cifar10 --weights cifar10_resnet34.pth
```

### 6.2 Verbose mode

```bash
neural-collapse resnet34 cifar10 --weights cifar10_resnet34.pth --verbose
```

### 6.3 Linear probes per layer

```bash
neural-collapse resnet34 cifar10 --weights cifar10_resnet34.pth --linear
```

### 6.4 Affine invariance test

```bash
neural-collapse resnet34 cifar10 --weights cifar10_resnet34.pth --affine
```

### 6.5 Training mode

```bash
neural-collapse resnet34 cifar10 --train ./checkpoints/resnet34.pth
```

---

## 7. CLI Behavior Summary

* `--weights` → inference with pretrained model
* `--train` → training mode
* `--linear` → linear separability analysis
* `--affine` → invariance analysis under linear transformations
* `--verbose` → detailed logs

---

## 8. Notes

* `linear` and `affine` can be used together.
* If neither `--weights` nor `--train` is provided, behavior depends on implementation.
* `start_layer` affects only analysis depth, not architecture.
* All generated plots are saved to the `plots/` directory.

```
