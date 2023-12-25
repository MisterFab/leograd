import sys
sys.path.append('C:/Users/leona/python/leograd')
from tensor import Tensor, CrossEntropyLoss
from nn import Module, Linear
from optim import SGD, Adam
from utils import EarlyStopping

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

def train_custom(x_train, y_train, x_val, y_val, batch_size=16):
    class CategoricalClassification(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(20, 64)
            self.linear2 = Linear(64, 5)

        def forward(self, x):
            x = self.linear1(x).relu()
            x = self.linear2(x)
            return x

    model = CategoricalClassification()
    epochs = 1000
    loss_function = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    for epoch in range(epochs):
        total_train_loss = 0

        for x_batch, y_batch in create_batches(x_train, y_train, batch_size):
            optimizer.zero_grad()
            preds = model(Tensor(x_batch, dtype=np.float32))
            loss = loss_function(preds, Tensor(y_batch, dtype=np.int32))
            total_train_loss += loss.data
            loss.backward()
            optimizer.step()
      
        total_val_loss = 0
        
        for x_batch, y_batch in create_batches(x_val, y_val, batch_size):
            preds = model(Tensor(x_batch, dtype=np.float32))
            loss = loss_function(preds, Tensor(y_batch, dtype=np.int32))
            total_val_loss += loss.data

        avg_train_loss = total_train_loss / (len(x_train) / batch_size)
        avg_val_loss = total_val_loss / (len(x_val) / batch_size)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        if early_stopping(avg_val_loss):
            print("Early stopping triggered at epoch", epoch)
            break

    return model

def train_pytorch(x_train, y_train, x_val, y_val, batch_size=64):
    class CategoricalClassification(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(20, 64)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(64, 5)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

    model = CategoricalClassification()

    epochs = 1000
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            preds = model(x_batch)
            loss = loss_function(preds, y_batch)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
    
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = model(x_batch)
                loss = loss_function(preds, y_batch)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / (len(x_train) / batch_size)
        avg_val_loss = total_val_loss / (len(x_val) / batch_size)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        if early_stopping(avg_val_loss):
            print("Early stopping triggered at epoch", epoch)
            break
    
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

x_data, y_data = make_classification(n_samples=10000, n_features=20, random_state=0, n_classes=5, n_informative=10)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

print("Custom implementation:")
now = time.time()
custom_model = train_custom(x_train, y_train, x_val, y_val)
print("Time:", time.time() - now)

print("\nPyTorch implementation:")
now = time.time()
pytorch_model = train_pytorch(x_train, y_train, x_val, y_val)
print("Time:", time.time() - now)

print("\nCustom accuracy:", test_model_custom(custom_model, x_val, y_val))
print("PyTorch accuracy:", test_model_pytorch(pytorch_model, x_val, y_val))