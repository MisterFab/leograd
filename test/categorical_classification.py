import sys
sys.path.append('C:/Users/leona/python/leograd')
from tensor import Tensor, CrossEntropy
from nn import Module, Dense
from optim import SGD

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_batches(x, y, batch_size=64):
    return [(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(x), batch_size)]

def train_custom(x_data, y_data, batch_size=64):
    class CategoricalClassification(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Dense(20, 64)
            self.linear2 = Dense(64, 5)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x
    
    model = CategoricalClassification()

    epochs = 100
    loss_function = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0

        for x_batch, y_batch in create_batches(x_data, y_data, batch_size):
            optimizer.zero_grad()
            preds = model(Tensor(x_batch, dtype=np.float32))
            loss = loss_function(preds, Tensor(y_batch, dtype=np.int32))
            total_loss += loss.data

            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", total_loss / (len(x_data) / batch_size))

    return model

def train_pytorch(x_data, y_data, batch_size=64):
    class CategoricalClassification(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(20, 64)
            self.linear2 = nn.Linear(64, 5)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    model = CategoricalClassification()

    epochs = 100
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    dataset = TensorDataset(torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()

            preds = model(x_batch)
            loss = loss_function(preds, y_batch)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", total_loss / len(loader))
    
    return model

def test_model_custom(model, x_test, y_test):
    preds = model(Tensor(x_test, dtype=np.float32)).data
    preds = np.argmax(preds, axis=1)
    return accuracy_score(y_test, preds)

def test_model_pytorch(model, x_test, y_test):
    with torch.no_grad():
        preds = model(torch.tensor(x_test, dtype=torch.float32))
        preds = torch.argmax(preds, axis=1).numpy()
    return accuracy_score(y_test, preds)

x_data, y_data = make_classification(n_samples=1000, n_features=20, random_state=0, n_classes=5, n_informative=10)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("Custom implementation:")
now = time.time()
custom_model = train_custom(x_train, y_train)
print("Time:", time.time() - now)

print("\nPyTorch implementation:")
now = time.time()
pytorch_model = train_pytorch(x_train, y_train)
print("Time:", time.time() - now)

print("\nCustom accuracy:", test_model_custom(custom_model, x_test, y_test))
print("PyTorch accuracy:", test_model_pytorch(pytorch_model, x_test, y_test))