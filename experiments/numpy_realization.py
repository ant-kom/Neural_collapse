import numpy as np

# Параметры
K = 10          # количество кластеров/центров
n = 1000        # размерность пространства
N = 10000       # количество выборок на каждый центр

np.random.seed(42)  # для воспроизводимости

# ========== (1) Генерация Z ==========
Z = np.random.randn(n, K)  # матрица n x K
print("=== (1) Матрица Z (первые 5 строк, 5 столбцов) ===")
print(Z[:5, :5])
print()

# ========== (2) Генерация X ==========
X_list = []
for k in range(K):
    mean = Z[:, k].reshape(-1, 1)  # среднее для k-го кластера
    # Генерируем N векторов, каждый размерности n
    # X_k будет размера n x N
    X_k = mean + np.random.randn(n, N)
    X_list.append(X_k)

X = np.hstack(X_list)  # объединяем по горизонтали -> n x (K*N)
print(f"=== (2) Матрица X сгенерирована, размер = {X.shape} ===")
print()

# ========== (3) Пересчет средних в Z ==========
for k in range(K):
    start_idx = k * N
    end_idx = (k + 1) * N
    Z[:, k] = np.mean(X[:, start_idx:end_idx], axis=1)

print("=== (3) Матрица Z после обновления (первые 5 строк, 5 столбцов) ===")
print(Z[:5, :5])
print()

# ========== (4) Построение матрицы A0 ==========
ZTZ = Z.T @ Z  # K x K
try:
    ZTZ_inv = np.linalg.inv(ZTZ)
except np.linalg.LinAlgError:
    # на случай вырожденности (маловероятно, но добавим псевдообратную)
    ZTZ_inv = np.linalg.pinv(ZTZ)

# (Z^T Z)^{-1} Z^T  размером K x n
M = ZTZ_inv @ Z.T  # K x n

# Добавляем нулевые строки снизу, чтобы получить n x n
top_part = np.vstack([M, np.zeros((n - K, n))])  # n x n

# Вторая часть: I_n - Z (Z^T Z)^{-1} Z^T (проектор на ортогональное дополнение)
I_n = np.eye(n)
P_perp = I_n - Z @ ZTZ_inv @ Z.T

A0 = top_part + P_perp
B0 = A0 @ Z

print("=== (4) Матрица A0 (первые 5 строк, первые 5 столбцов) ===")
print(A0[:5, :5])
print(B0[:5, :5])

# ========== (6) Выборочная ковариация A0 * X ==========
Y = A0 @ X  # размер n x (K*N)
# Выборочная ковариационная матрица столбцов Y (размер n x n)
Sigma = np.cov(Y, bias=False)  # bias=False для выборочной (деление на (m-1))
print(f"=== (6) Матрица Sigma (n x n), отображаем угловой блок 5x5 ===")
print(Sigma[:5, :5])
print()

# ========== (7)-(8) Разбиение Sigma на блоки ==========
Sigma11 = Sigma[:K, :K]          # K x K
Sigma12 = Sigma[:K, K:]          # K x (n-K)
Sigma21 = Sigma[K:, :K]          # (n-K) x K
Sigma22 = Sigma[K:, K:]          # (n-K) x (n-K)

# ========== (9) Вычисление S и mu ==========
try:
    Sigma22_inv = np.linalg.inv(Sigma22)
except np.linalg.LinAlgError:
    Sigma22_inv = np.linalg.pinv(Sigma22)

S = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21

# e — вектор из K единиц
e = np.ones(K)

# ||S e||^2
Se = S @ e
norm_Se_sq = Se @ Se

# tr(S)
trace_S = np.trace(S)

# mu
mu = trace_S - norm_Se_sq / (e @ Se)

print("=== (9) Матрица S (первые 5x5) ===")
print(S[:5, :5])
print()

print(f"=== (10) mu = {mu} ===")
print()

# ========== (11) Генерация A (2n x n) и X = A * X ==========
A = np.random.randn(2 * n, n)
A += np.abs(A.min())
a = np.random.randn(2 * n, 1)
#A = np.random.uniform(low=0, high=1, size=(2 * n, n))   # (2n x n)
#a = np.random.uniform(low=0, high=1, size=(2 * n, 1))   # (2n x n)
X = A @ X + a.reshape(-1, 1)                        # теперь X размера (2n) x (K*N)
Z = A @ Z + a.reshape(-1, 1)                       # теперь X размера (2n) x (K*N)

print(f"=== (11) Матрица A сгенерирована, X обновлен, новый размер X = {X.shape} ===")
print(f"AMAX = {A.max()}")
print(f"AMIN = {A.min()}")
#Sing_A = np.linalg.svd(A, compute_uv=False)
#print(Sing_A[:5], Sing_A[-5:])
print()

# ========== (12) Повторяем шаги (3)--(10) ==========

# (3) снова пересчитываем Z как средние столбцов X
n_new = 2 * n  # новая размерность
K = K          # то же количество кластеров
for k in range(K):
    start_idx = k * N
    end_idx = (k + 1) * N
    Z[:, k] = np.mean(X[:, start_idx:end_idx], axis=1)

print("=== (12.3) Матрица Z после повторного обновления (первые 5 строк, 5 столбцов) ===")
print(Z[:5, :5])
print()

# (4) новая A0
ZTZ = Z.T @ Z
ZTZ_inv = np.linalg.inv(ZTZ) if np.linalg.det(ZTZ) > 1e-12 else np.linalg.pinv(ZTZ)
M = ZTZ_inv @ Z.T
top_part = np.vstack([M, np.zeros((n_new - K, n_new))])
P_perp = np.eye(n_new) - Z @ ZTZ_inv @ Z.T
A0 = top_part + P_perp

print("=== (12.4) Новая матрица A0 (первые 5 строк, первые 5 столбцов) ===")
print(A0[:5, :5])
print()

# (6) выборочная ковариация
Y = A0 @ X
Sigma = np.cov(Y, bias=False)

print("=== (12.6) Новая Sigma (первые 5x5) ===")
print(Sigma[:5, :5])
print()

# (7)-(8) разбиение
Sigma11 = Sigma[:K, :K]
Sigma12 = Sigma[:K, K:]
Sigma21 = Sigma[K:, :K]
Sigma22 = Sigma[K:, K:]

# (9) S и mu
Sigma22_inv = np.linalg.inv(Sigma22) if np.linalg.det(Sigma22) > 1e-12 else np.linalg.pinv(Sigma22)
S = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21
Se = S @ e
mu1 = np.trace(S) - (Se @ Se) / (e @ Se)

print("=== (12.9) Новая матрица S (первые 5x5) ===")
print(S[:5, :5])
print()

print(f"=== (12.10) Новое mu = {mu1} ===")

print(f"=== (12.10) Старое mu = {mu} ===")