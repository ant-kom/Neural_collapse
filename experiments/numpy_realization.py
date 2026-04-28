import numpy as np


n = 1000
K = 10
N = 80     # критически малое N

d = 2      # сверхнизкий ранг (почти вырождение)

# скрытое подпространство
U = np.random.randn(n, d)
U = U / np.linalg.norm(U, axis=0, keepdims=True)

Z = np.random.randn(n, K)

X_list = []

for k in range(K):
    mean = Z[:, k].reshape(-1, 1)

    # общий латентный фактор (КЛЮЧЕВО: одинаковый для всех координат)
    latent = np.random.randn(d, N)

    # сильная корреляция всех признаков через одно подпространство
    X_k = U @ latent

    # почти нулевой шум (чтобы сохранить вырожденность)
    X_k += 1e-6 * np.random.randn(n, N)

    X_k += mean

    X_list.append(X_k)

X = np.hstack(X_list)
print(f"=== (2) Матрица X, размер = {X.shape} ===")
print()

# ========== (3) Пересчёт средних в Z ==========
for k in range(K):
    start_idx = k * N
    end_idx = (k + 1) * N
    Z[:, k] = np.mean(X[:, start_idx:end_idx], axis=1)

print("=== (3) Матрица Z после обновления (первые 5 строк, 5 столбцов) ===")
print(Z[:5, :5])
print()

# ========== (4) Построение матрицы A0 ==========
ZTZ = Z.T @ Z
try:
    ZTZ_inv = np.linalg.inv(ZTZ)
except np.linalg.LinAlgError:
    ZTZ_inv = np.linalg.pinv(ZTZ)

M = ZTZ_inv @ Z.T  # K x n

top_part = np.vstack([M, np.zeros((n - K, n))])  # n x n

I_n = np.eye(n)
P_perp = I_n - Z @ ZTZ_inv @ Z.T

A0 = top_part + P_perp
B0 = A0 @ Z

print("=== (4) Матрица A0 (первые 5x5) ===")
print(A0[:5, :5])
print(B0[:5, :5])

# ========== (6) Выборочная ковариация A0 * X ==========
Y = A0 @ X
Sigma = np.cov(Y, bias=False)

print("=== (6) Матрица Sigma (5x5) ===")
print(Sigma[:5, :5])
print()

# ========== (7)-(8) Разбиение Sigma ==========
def schur_complement_robust(Sigma, K, tol=1e-10):
    n = Sigma.shape[0]
    Sigma11 = Sigma[:K, :K]
    Sigma12 = Sigma[:K, K:]
    Sigma21 = Sigma[K:, :K]
    Sigma22 = Sigma[K:, K:]

    U, s, Vt = np.linalg.svd(Sigma22, hermitian=True)

    s_max = s[0] if len(s) > 0 else 0
    rank = np.sum(s > tol * s_max)

    if rank == len(s):
        Sigma22_inv = np.linalg.inv(Sigma22)
        S = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21
    else:
        U1 = U[:, :rank]

        Sigma22_proj = U1.T @ Sigma22 @ U1
        Sigma22_proj_inv = np.linalg.inv(Sigma22_proj)

        Sigma12_proj = Sigma12 @ U1

        S = Sigma11 - Sigma12_proj @ Sigma22_proj_inv @ Sigma12_proj.T

    return S

S = schur_complement_robust(Sigma, K, tol=1e-10)

e = np.ones(K)

Se = S @ e
norm_Se_sq = Se @ Se

trace_S = np.trace(S)

mu = trace_S - norm_Se_sq / (e @ Se)

print("=== (9) Матрица S (5x5) ===")
print(S[:5, :5])
print(f"tr(S) = {trace_S}")
print(f"||Se||^2 = {norm_Se_sq}")
print(f"e^T S e = {e @ Se}")

print(f"=== (10) mu = {mu} ===")
print()

# ========== (11) Генерация A (2n x n) ==========
A = np.zeros((2 * n, n))

# верхний блок — усиливает первые координаты
A[:n, :n] = np.diag(np.linspace(5.0, 0.1, n))

# нижний блок — смешивание координат (жёсткая неортогональность)
B = np.random.randn(n, n)
A[n:, :n] = B @ np.diag(np.linspace(0.1, 5.0, n))

# сдвиг (усиливает bias)
a = np.random.randn(2 * n, 1) * 0.1

X = A @ X + a.reshape(-1, 1)
Z = A @ Z + a.reshape(-1, 1)

print(f"=== (11) Матрица A создана, новый размер X = {X.shape} ===")
print()

# ========== (12) Повтор шагов ==========
n = 2 * n

for k in range(K):
    start_idx = k * N
    end_idx = (k + 1) * N
    Z[:, k] = np.mean(X[:, start_idx:end_idx], axis=1)

print("=== (12.3) Новая Z (5x5) ===")
print(Z[:5, :5])
print()

ZTZ = Z.T @ Z
ZTZ_inv = np.linalg.inv(ZTZ) if np.linalg.det(ZTZ) > 1e-12 else np.linalg.pinv(ZTZ)

M = ZTZ_inv @ Z.T
top_part = np.vstack([M, np.zeros((n - K, n))])
P_perp = np.eye(n) - Z @ ZTZ_inv @ Z.T

A0 = top_part + P_perp

print("=== (12.4) Новая A0 (5x5) ===")
print(A0[:5, :5])
print()

Y = A0 @ X
Sigma = np.cov(Y, bias=False)

print("=== (12.6) Новая Sigma (5x5) ===")
print(Sigma[:5, :5])
print()

S = schur_complement_robust(Sigma, K, tol=1e-10)

Se = S @ e
mu1 = np.trace(S) - (Se @ Se) / (e @ Se)

print("=== (12.9) Новая S (5x5) ===")
print(S[:5, :5])
print()

print(f"=== (12.10) Новое mu = {mu1} ===")
print(f"=== (12.10) Старое mu = {mu} ===")