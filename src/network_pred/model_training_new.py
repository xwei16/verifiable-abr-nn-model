import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm   # progress bar
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from pathlib import Path
from utils import load_network_data

'''
Neural Network Model
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)  # Hidden layer 1
        self.fc3 = nn.Linear(64, 1)    # Output layer
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))  # Input -> Hidden 1
        x = self.relu(self.bn2(self.fc2(x)))  # Hidden 1 -> Hidden 2
        x = self.fc3(x)                        # Hidden 2 -> Output
        return x

"""
MAPE loss that directly optimizes relative error
epsilon prevents division by zero
"""
def mape_loss(output, target, epsilon=1e-8):
    return torch.mean(torch.abs((target - output) / (target + epsilon)))



'''
Training loop with progress bar
'''
def train_model(X_train, y_train, epochs=200, batch_size=64, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Normalize inputs
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std

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

    return model, X_mean, X_std, y_mean, y_std

"""
Save normalization parameters to a file
"""
def save_normalization_params(X_mean, X_std, y_mean, y_std, filepath='normalization_params.npz'):
    
    np.savez(filepath, 
             X_mean=X_mean, 
             X_std=X_std, 
             y_mean=y_mean, 
             y_std=y_std)
    print(f"Normalization parameters saved to {filepath}")
    
'''
Example usage
'''
if __name__ == "__main__":
    X_train, y_train = load_network_data("../../puffer_data_cleaned/training_data", 100)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    model, X_mean, X_std, y_mean, y_std = train_model(X_train, y_train, epochs=500, batch_size=32, lr=1e-2)
    torch.save(model, "output/nn_network_model_10days_100_data.pt")
    save_normalization_params(X_mean, X_std, y_mean, y_std, filepath='normalization_params.npz')