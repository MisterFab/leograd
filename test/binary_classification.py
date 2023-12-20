import sys
sys.path.append('C:/Users/leona/python/leograd')
from tensor import Tensor, BinaryCrossEntropy
from nn import Module, Dense
from optim import SGD

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def train_custom(x_data, y_data):
    class BinaryClassification(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Dense(10, 64)
            self.linear2 = Dense(64, 1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x.sigmoid()
    
    model = BinaryClassification()

    epochs = 100
    loss_function = BinaryCrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(Tensor(x_data, dtype=np.float32))
        loss = loss_function(preds, Tensor(y_data, dtype=np.float32))

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.data)

        loss.backward()
        optimizer.step()

def train_pytorch(x_data, y_data):
    class BinaryClassification(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 64)
            self.linear2 = nn.Linear(64, 1)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return torch.sigmoid(x)

    model = BinaryClassification()

    epochs = 100
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(torch.tensor(x_data, dtype=torch.float32))
        loss = loss_function(preds, torch.tensor(y_data, dtype=torch.float32))
        
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.item())

        loss.backward()
        optimizer.step()

x_data, y_data = make_classification(n_samples=100, n_features=10, random_state=0)
y_data = y_data.reshape(-1, 1)

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

print("Custom implementation:")
now = time.time()
train_custom(x_data, y_data)
print("Time:", time.time() - now)

print("\nPyTorch implementation:")
now = time.time()
train_pytorch(x_data, y_data)
print("Time:", time.time() - now)