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


# Define Net in this file so torch.load can find it
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

"""
Load normalization parameters from a file
"""
def load_normalization_params(filepath='normalization_params.npz'):
    data = np.load(filepath)
    X_max = data['X_max']
    y_max = data['y_max']
    print(f"Normalization parameters loaded from {filepath}")
    return X_max, y_max


'''
Clopper-Pearson evaluation
'''
def clopper_pearson_eval(model, X_test, y_test, eps_tolerance=0.01, alpha=0.05, output_file="output/pred_vs_true.csv", normalized_param_file="normalization_params.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_max, y_max = load_normalization_params(normalized_param_file)
    print(f"=== BEFORE NORMALIZATION ===")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    X_test = X_test / X_max
    print(f"\n=== AFTER NORMALIZATION ===")
    print(f"X_test shape: {X_test.shape}")
    y_test = y_test / y_max
    print(f"y_test shape: {y_test.shape}")
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        # print(f"\n=== TENSOR CONVERSION ===")
        # print(f"X_test_t shape: {X_test_t.shape}")

        y_pred_norm = model(X_test_t)

        # print(f"\n=== MODEL OUTPUT ===")
        # print(f"y_pred_norm shape (raw): {y_pred_norm.shape}")
        # print(f"y_pred_norm first 5 values: {y_pred_norm[:5].cpu().numpy()}")

        y_pred_norm = y_pred_norm.cpu().numpy().reshape(-1)
        # print(f"\n=== AFTER CPU/NUMPY ===")
        # print(f"y_pred_norm_np shape: {y_pred_norm.shape}")
        
        y_pred_norm.flatten()

        # print(f"\n=== AFTER FLATTEN ===")
        # print(f"y_pred_norm_flat shape: {y_pred_norm.shape}")
        # print(f"y_pred_norm_flat first 5: {y_pred_norm[:5]}")
        
        # y_pred = y_pred_norm * y_std + y_mean

        # print(f"\n=== FINAL DENORMALIZATION ===")
        # print(f"y_pred shape: {y_pred.shape}")
        # print(f"y_test shape: {y_test.shape}")
        # print(f"y_pred first 5: {y_pred[:5]}")
        # print(f"y_test first 5: {y_test[:5]}")

    # Put them into a DataFrame
    df = pd.DataFrame({
        "true_f(x)": y_test,
        "predicted_f'(x)": y_pred_norm,
        "relative_error": np.abs(y_test - y_pred_norm) # / (y_test + 1e-8)
    })

    # Save to CSV file
    df.to_csv(output_file, index=False)

    # Success = relative error ≤ eps
    successes = np.abs((y_pred_norm - y_test)) <= eps_tolerance #  / (y_test + 1e-8))
    
    k = np.sum(successes)
    n = len(y_test)
    p_hat = k / n

    ci_low, ci_high = proportion_confint(k, n, alpha=alpha, method='beta')

    print("\n=== Clopper-Pearson Analysis ===")
    print(f"success = {k}, total = {n}")
    print(f"Observed probability (relative error ≤ {eps_tolerance}): {p_hat:.4f}")
    print(f"{int((1-alpha)*100)}% CI: [{ci_low:.4f}, {ci_high:.4f}]")


'''
Load model + evaluate
'''
if __name__ == "__main__":
    X_train, y_train, _ = load_network_data_per_file("../../data/puffer/puffer_data_cleaned/training_data", start_row=1, nrows=1000)
    X_test, y_test, _ = load_network_data_per_file("../../data/puffer/puffer_data_cleaned/training_data", start_row=1001, nrows=1000)
    for i in range(len(X_train)):
        # Load saved model
        model = torch.load(f"experiments/3-layer/nn_network_model_{i}.pt")

        print(f"Training data shape: X={X_train[i].shape}, y={y_train[i].shape}")
        
        
        # Run Clopper-Pearson
        clopper_pearson_eval(model, X_train[i], y_train[i], eps_tolerance=0.1, alpha=0.1, 
                            output_file=f"experiments/3-layer/training_pred_vs_true_{i}.csv", normalized_param_file=f"experiments/3-layer/normalization_params{i}.npz")
        clopper_pearson_eval(model, X_test[i], y_test[i], eps_tolerance=0.1, alpha=0.1, 
                            output_file=f"experiments/3-layer/training_pred_vs_true_{i}.csv", normalized_param_file=f"experiments/3-layer/normalization_params{i}.npz")
       