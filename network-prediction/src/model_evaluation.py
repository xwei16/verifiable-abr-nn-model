import numpy as np
import pandas as pd
import pickle
from statsmodels.stats.proportion import proportion_confint
import torch
import torch.nn as nn
from utils import load_network_data

# Define Net in this file so torch.load can find it
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
Load normalization parameters from a file
"""
def load_normalization_params(filepath='normalization_params.npz'):
    data = np.load(filepath)
    X_mean = data['X_mean']
    X_std = data['X_std']
    y_mean = data['y_mean']
    y_std = data['y_std']
    print(f"Normalization parameters loaded from {filepath}")
    return X_mean, X_std, y_mean, y_std


'''
Clopper-Pearson evaluation
'''
def clopper_pearson_eval(model, X_test, y_test, eps_tolerance=0.01, alpha=0.05, output_file="output/pred_vs_true.csv", normalized_param_file="normalization_params.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_mean, X_std, y_mean, y_std = load_normalization_params(normalized_param_file)
    print(f"=== BEFORE NORMALIZATION ===")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    X_test = (X_test - X_mean) / X_std
    print(f"\n=== AFTER NORMALIZATION ===")
    print(f"X_test shape: {X_test.shape}")

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
        
        y_pred = y_pred_norm * y_std + y_mean

        # print(f"\n=== FINAL DENORMALIZATION ===")
        # print(f"y_pred shape: {y_pred.shape}")
        # print(f"y_test shape: {y_test.shape}")
        # print(f"y_pred first 5: {y_pred[:5]}")
        # print(f"y_test first 5: {y_test[:5]}")

    # Put them into a DataFrame
    df = pd.DataFrame({
        "true_f(x)": y_test,
        "predicted_f'(x)": y_pred,
        "relative_error": np.abs(y_test - y_pred) / (y_test + 1e-8)
    })

    # Save to CSV file
    df.to_csv(output_file, index=False)

    # Success = relative error ≤ eps
    successes = np.abs((y_pred - y_test) / (y_test + 1e-8)) <= eps_tolerance
    
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
    # Load saved model
    model = torch.load("output-models/nn_network_model_10days_100_data.pt")
    #"output/10000_rows_trained_2layers_model/nn_network_model_10days_1000_data.pt"
    # Load data (same file or a new test set)
    X_train, y_train = load_network_data("../../data/puffer/puffer_data_cleaned/training_data", nrows=100)
    X_test, y_test = load_network_data("../../data/puffer/puffer_data_cleaned/testing_data") 
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Run Clopper-Pearson
    clopper_pearson_eval(model, X_train, y_train, eps_tolerance=0.3, alpha=0.10, 
                        output_file="output-models/1000_rows_trained_2layers_L1_model/training_pred_vs_true.csv", normalized_param_file="normalization_params.npz")
    clopper_pearson_eval(model, X_test, y_test, eps_tolerance=0.3, alpha=0.10, 
                        output_file="output-models/1000_rows_trained_2layers_L1_model/testing_pred_vs_true.csv", normalized_param_file="normalization_params.npz")

    # Feature ranges analysis
    # Get column names for better readability
    # df_train = pd.read_csv("output/network_data_training_1000_new.txt")
    # df_test = pd.read_csv("output/network_data_testing_1000_new.txt")
    # feature_names = [col for col in df_train.columns if col != 'downloading_time_9']
    
    # print("\n=== Feature Ranges ===")
    # for i, name in enumerate(feature_names):
    #     print(f"\n{name}:")
    #     print(f"  Training: {X_train[:, i].min():.6f} to {X_train[:, i].max():.6f}")
    #     print(f"  Testing:  {X_test[:, i].min():.6f} to {X_test[:, i].max():.6f}")

    # Theoretical baseline using the 9th chunk's values
    # Extract indices for bandwidth (index 16), delay (index 17), chunk_size_9 (index 18)
    # bandwidth_idx = feature_names.index('bandwidth')
    # delay_idx = feature_names.index('delay')
    # chunk_size_9_idx = feature_names.index('chunk_size_9')
    
    # y_theory = X_test[:, chunk_size_9_idx] / X_test[:, bandwidth_idx] + X_test[:, delay_idx]
    # theory_errors = np.abs(y_theory - y_test) / (y_test + 1e-8)
    # theory_success = (theory_errors <= 0.2).sum()
    # print(f"\n=== Theoretical Baseline ===")
    # print(f"Formula: downloading_time = chunk_size_9 / bandwidth + delay")
    # print(f"Success rate: {theory_success}/{len(y_test)} = {theory_success/len(y_test)*100:.1f}%")