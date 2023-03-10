import numpy as np
import pandas as pd

# Загрузка данных из файла в массив
data = np.loadtxt('linreg_4.txt')
# Разделение данных на матрицу признаков X и вектор-столбец целевой переменной y
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

# Нормализация признаков
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Добавление столбца единиц в матрицу X для учета свободного коэффициента
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Инициализация начальных значений параметров
theta = np.zeros((X.shape[1], 1))
alpha = 0.01
iterations = 1000