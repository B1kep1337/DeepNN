# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import Perceptron

df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes1 = 10      # задаем число нейронов первого скрытого слоя 
hiddenSizes2 = 10      # задаем число нейронов второго скрытого слоя
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов

NN = Perceptron(inputSize, hiddenSizes1, hiddenSizes2, outputSize)

NN.train(X, y, n_iter=5, eta=0.01)

# Оценка на всём датасете
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values

outs = []
for xi in X:
    out, h1, h2 = NN.predict(xi)
    outs.append(out)

outs = np.array(outs)

# ====== ДОБАВЛЕННЫЙ БЛОК РАСЧЁТА ТОЧНОСТИ ======
# 1. Приводим массивы к одномерному виду, чтобы избежать ошибок broadcast'а
outs_flat = outs.flatten()
y_flat = y.flatten()

# 2. Считаем количество совпадений и переводим в проценты
correct_predictions = np.sum(outs_flat == y_flat)
accuracy = (correct_predictions / len(y_flat)) * 100

print(f" Точность модели: {accuracy:.2f}% ({correct_predictions} из {len(y_flat)} верных предсказаний)")