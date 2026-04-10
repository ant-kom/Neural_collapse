# Алгоритм вычисления метрики

## Входные данные
- Модель классификации `model` (выход: `(batch_size, K)`)
- Название слоя `layer_name`
- Даталоадер `dataloader`

---

## Обозначения
- $n$ — размерность признаков  
- $K$ — число классов  
- $X \in \mathbb{R}^n$ — вектор признаков  
- $Z \in \mathbb{R}^{n \times K}$ — матрица условных ожиданий  
- $Var(X) \in \mathbb{R}^{n \times n}$ — ковариация  

---

## Шаг 0. Определение числа классов

Пропустить один батч через модель:

$$
K = \text{output.shape}[1]
$$

---

## Шаг 1. Оценка матрицы $Z$

Для каждого класса $k$:

$$
Z[:, k] = \mathbb{E}[X \mid Y = k]
$$

Итог:

$$
Z =
\begin{bmatrix}
\mu_0 & \mu_1 & \dots & \mu_{K-1}
\end{bmatrix}
\in \mathbb{R}^{n \times K}
$$

---

## Шаг 2. Матрица $A_0$

Проектор:

$$
P = Z (Z^T Z)^{-1} Z^T
$$

Блочная матрица:

$$
B =
\begin{bmatrix}
(Z^T Z)^{-1} Z^T \\
0
\end{bmatrix}
$$

Итог:

$$
A_0 = B + I - P
$$

---

## Шаг 3. Ковариация

$$
Var(X) = \mathbb{E}[(X - \mathbb{E}X)(X - \mathbb{E}X)^T]
$$

$$
\Sigma = A_0 \cdot Var(X) \cdot A_0^T
$$

---

## Шаг 4. Блочное разбиение

$$
\Sigma =
\begin{bmatrix}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{bmatrix}
$$

Размерности:
- $\Sigma_{11} \in \mathbb{R}^{K \times K}$
- $\Sigma_{12} \in \mathbb{R}^{K \times (n-K)}$
- $\Sigma_{21} \in \mathbb{R}^{(n-K) \times K}$
- $\Sigma_{22} \in \mathbb{R}^{(n-K) \times (n-K)}$

---

## Шаг 5. Матрица $S$

$$
S = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}
$$

---

## Шаг 6. Метрика

$$
\mathbf{1} = (1, \dots, 1)^T
$$

$$
\text{Metric} =
\operatorname{tr}(S) - \frac{1}{K} \mathbf{1}^T S \mathbf{1}
$$

---

## Выход
Скалярное значение метрики

---

