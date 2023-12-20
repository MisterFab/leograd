import sys
sys.path.append('C:/Users/leona/python/leograd')
from tensor import Tensor, MeanSquaredError
from nn import Module, Dense
from optim import SGD

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def train_custom(x_data, y_data):
    class LinearRegression(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Dense(10, 32)
            self.linear2 = Dense(32, 1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x
    
    model = LinearRegression()

    epochs = 100
    loss_function = MeanSquaredError()
    optimizer = SGD(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(Tensor(x_data, dtype=np.float32))
        loss = loss_function(preds, Tensor(y_data, dtype=np.float32))

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.data)

        loss.backward()
        optimizer.step()

def train_pytorch(x_data, y_data):
    class LinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 32)
            self.linear2 = nn.Linear(32, 1)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    model = LinearRegression()

    epochs = 100
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(torch.tensor(x_data, dtype=torch.float32))
        loss = loss_function(preds, torch.tensor(y_data, dtype=torch.float32))
 
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.item())

        loss.backward()
        optimizer.step()

x_data, y_data = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=0)
y_data = y_data.reshape(-1, 1)

scaler = StandardScaler()

X_standardized = scaler.fit_transform(x_data)

print("Custom implementation:")
now = time.time()
train_custom(X_standardized, y_data)
print("Time:", time.time() - now)

print("\nPyTorch implementation:")
now = time.time()
train_pytorch(X_standardized, y_data)
print("Time:", time.time() - now)