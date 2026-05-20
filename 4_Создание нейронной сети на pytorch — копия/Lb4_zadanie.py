import numpy as np
import pandas as pd
import torch
from torch import nn

n = 19
if (n % 2) == 1:
    print('Решите задачу классификации покупателей на классы купит - не купит (3й столбец) по признакам возраст и доход.')
    
    df = pd.read_csv('dataset_simple.csv')
    X = torch.tensor(df.iloc[:, 0:2].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, 2].values.reshape(-1, 1), dtype=torch.float32)
    
    # Стандартизация
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)

    class NNet_binary(nn.Module):
        def __init__(self, in_size, hidden_size, out_size):
            super().__init__()
            # nn.Sequential - контейнер модулей
            # он последовательно объединяет слои и позволяет запускать их одновременно
            self.layers = nn.Sequential(
                nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                nn.Tanh(),
                nn.Linear(hidden_size, out_size),
                nn.Sigmoid() # Выход в диапазоне [0, 1]
            )
        def forward(self, x):
            return self.layers(x)

    inputSize = X.shape[1]
    hiddenSize = 10
    outputSize = 1
    
    net = NNet_binary(inputSize, hiddenSize, outputSize)
    
    # 1. Используем BCELoss для бинарной классификации
    lossFn = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    epochs = 1000

    for i in range(epochs):
        optimizer.zero_grad()    #обнуляем градиенты
        pred = net(X)            #делаем прямой проход для выдачи предсказания
        loss = lossFn(pred, y)   #вычисление ошибки
        loss.backward()          #обратное вычисление, вычисляем градиенты
        optimizer.step()         #обновление весов
        
        if (i + 1) % 100 == 0:
            print(f'Ошибка на {i + 1} итерации: {loss.item():.4f}')

    # Оценка качества
    with torch.no_grad():
        pred = net(X)
        # 2. Правильный порог для сигмоиды: 0.5
        pred_labels = (pred >= 0.5).float()
        
        # 3. Считаем количество неверных предсказаний напрямую
        num_errors = torch.sum(pred_labels != y).item()
        print(f'\nКоличество ошибок: {num_errors} из {len(y)}')
        print(f'Точность: {(1 - num_errors/len(y)) * 100:.2f}%')