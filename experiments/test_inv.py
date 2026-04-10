import torch

n = 30000

A = torch.randn((n, n), device="cuda")

torch.linalg.pinv(A)