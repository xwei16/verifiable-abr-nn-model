from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm   # progress bar
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from pathlib import Path
import os
os.makedirs("output-models/experiments", exist_ok=True)
os.makedirs("output-models/experiments/norms", exist_ok=True)

def load_network_data_per_file(dir_path, start_row=0, nrows=None):
    dir_path = Path(dir_path)
    csv_files = sorted(dir_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {dir_path}")

    print(f"Found {len(csv_files)} CSV files")

    X_list = []
    y_list = []
    file_names = []

    for csv_file in csv_files:
        print(f"Loading: {csv_file.name} with {nrows} rows")
        
        df = pd.read_csv(csv_file, skiprows=range(1, start_row+1), nrows=nrows)

        X = df.drop(columns=["downloading_time_9"]).values
        y = df["downloading_time_9"].values

        X_list.append(X)
        y_list.append(y)
        file_names.append(csv_file.name)

    return X_list, y_list, file_names


'''
Neural Network Model
'''
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(19, 128)  # Input layer
#         self.fc2 = nn.Linear(128, 64)  # Hidden layer 1
#         self.fc3 = nn.Linear(64, 1)    # Output layer
        
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(64)
        
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.relu(self.bn1(self.fc1(x)))  # Input -> Hidden 1
#         x = self.relu(self.bn2(self.fc2(x)))  # Hidden 1 -> Hidden 2
#         x = self.fc3(x)                        # Hidden 2 -> Output
#         return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # --- Hidden layers ---
        self.fc1 = nn.Linear(19, 128)   # Hidden layer 1
        self.fc2 = nn.Linear(128, 64)   # Hidden layer 2
        self.fc3 = nn.Linear(64, 32)    # Hidden layer 3
        self.fc4 = nn.Linear(32, 1)     # Output layer

        # --- BatchNorm ---
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))  # Layer 1
        x = self.relu(self.bn2(self.fc2(x)))  # Layer 2
        x = self.relu(self.bn3(self.fc3(x)))  # Layer 3
        x = self.fc4(x)                        # Output
        return x

'''
Training loop with progress bar
'''
def train_model(X_train, y_train, epochs=200, batch_size=64, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Normalize inputs
    # X_mean = X_train.mean(axis=0)
    # X_std = X_train.std(axis=0) + 1e-8
    # X_train = (X_train - X_mean) / X_std
    # y_mean = y_train.mean()
    # y_std = y_train.std()
    # y_train = (y_train - y_mean) / y_std
    
    
    # Normalize inputs by max absolute value
    X_max = np.max(np.abs(X_train), axis=0) + 1e-8
    X_train = X_train / X_max

    # Normalize labels by max absolute value
    y_max = np.max(np.abs(y_train)) + 1e-8
    y_train = y_train / y_max

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Net().to(device)

    # Use L1Loss for better balance with varying magnitude targets
    criterion = nn.L1Loss()
    # criterion = mape_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return model, X_max, y_max

"""
Save normalization parameters to a file
"""
def save_normalization_params(X_max, y_max, filepath='normalization_params.npz'):
    
    np.savez(filepath, 
             X_max=X_max, 
             y_max=y_max)
    print(f"Normalization parameters saved to {filepath}")

'''
Example usage
'''
if __name__ == "__main__":
    X_train, y_train, file_names = load_network_data_per_file("../../data/puffer/puffer_data_cleaned/training_data", 1, 1000)
    for i in range(len(X_train)):
        print(f"File: {file_names[i]}, Data shape: X={X_train[i].shape}, y={y_train[i].shape}")
        model, X_max, y_max = train_model(X_train[i], y_train[i], epochs=500, batch_size=32, lr=1e-2)
        torch.save(model, f"experiments/3-layer/nn_network_model_{i}.pt")
        save_normalization_params(X_max, y_max, filepath=f'experiments/3-layer/normalization_params{i}.npz')
        
    # print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    