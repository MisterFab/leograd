import sys
sys.path.append('C:/Users/leona/python/leograd')
from tensor import Tensor, MeanSquaredError
from nn import Module, Dense
from optimizer import SGD

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.datasets import make_regression

def train_custom(x_data, y_data):
    class LinearRegression(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Dense(5, 10)
            self.linear2 = Dense(10, 1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x
    
    model = LinearRegression()

    epochs = 100
    loss_function = MeanSquaredError()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(Tensor(x_data))
        loss = loss_function(preds, Tensor(y_data))

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.data)

        loss.backward()
        optimizer.step()

def train_pytorch(x_data, y_data):
    class LinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(5, 10)
            self.linear2 = nn.Linear(10, 1)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    model = LinearRegression()

    epochs = 100
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(torch.tensor(x_data, dtype=torch.float32))
        loss = loss_function(preds, torch.tensor(y_data, dtype=torch.float32))
 
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.item())

        loss.backward()
        optimizer.step()

x_data, y_data = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=0)
y_data = y_data.reshape(-1, 1)

scaler = StandardScaler()

X_standardized = scaler.fit_transform(x_data)
y_standardized = scaler.fit_transform(y_data)

print("Custom implementation:")
now = time.time()
train_custom(X_standardized, y_standardized)
print("Time:", time.time() - now)
print("\nPyTorch implementation:")
now = time.time()
train_pytorch(X_standardized, y_standardized)
print("Time:", time.time() - now)