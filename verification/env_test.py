import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from utils import load_network_data


# =========================
# Model Definition
# =========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# =========================
# Load normalization params
# =========================
def load_normalization_params(filepath):
    data = np.load(filepath)
    return data['X_max'], data['y_max']


# =========================
# Compute LiRPA Bounds
# =========================
def compute_dataset_lirpa_bound(f, model, X, X_max, y_max, method='backward'):

    # X_max type: <class 'numpy.ndarray'>
    # y_max type: <class 'numpy.ndarray'>
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Normalize
    X_norm = X / X_max
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    # Build global box from dataset
    lb = X_tensor.min(dim=0).values.unsqueeze(0)
    ub = X_tensor.max(dim=0).values.unsqueeze(0)
    
    print(lb, ub)

    f.write(
        f"input bounds:\n"
        f"lower: {lb.cpu().numpy() * X_max}\n"
        f"upper: {ub.cpu().numpy()  * X_max}\n"
    )

    f.write(
        f"input bounds (norm):\n"
        f"lower: {lb}\n"
        f"upper: {ub}\n"
    )
    # Wrap model
    dummy = torch.zeros(1, X_tensor.shape[1]).to(device)
    lirpa_model = BoundedModule(model, dummy, device=device)

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    center = (lb + ub) / 2.0
    x = BoundedTensor(center, ptb)

    lb_out, ub_out = lirpa_model.compute_bounds(x=(x,), method=method)
    f.write(
        f"dt(norm) bounds:\n"
        f"lower: {lb_out}\n"
        f"upper: {ub_out}\n"
    )
    lb_out = max(0, lb_out.item() * y_max)
    ub_out = ub_out.item() * y_max
    f.write(
        f"dt bounds:\n"
        f"lower: {lb_out}\n"
        f"upper: {ub_out}\n"
    )

    return lb_out, ub_out


# =========================
# Main
# =========================
if __name__ == "__main__":

    # Load trained model
    model = torch.load(
        "../model/network-prediction-model/network_pred.pt",
        weights_only=False
    )

    # Load normalization parameters
    X_max, y_max = load_normalization_params(
        "../model/network-prediction-model/normalization_params.npz"
    )
    
    # Load test data
    X_test, y_test = load_network_data(
        "../data/puffer/puffer_data_cleaned/testing_data",
        nrows=50
    )
    print(X_test)

    print("Computing certified bounds...")
    
    f = open("logs/env_bounds.log", "w")
    f.write(f"{X_max}")
    f.write(f"{y_max}")
    # Compute bounds
    for idx, X in enumerate(X_test):
        f.write(f"--- Dataset {idx} ---\n")
        methods = ["IBP", "CROWN", "CROWN-Optimized"]
        for method in methods:
            f.write(f"--- {method} ---\n")
            lb_out, ub_out = compute_dataset_lirpa_bound(
                f,
                model,
                X,
                X_max,
                y_max,
                method=method,   # options: 'IBP', 'forward', 'backward', 'CROWN-IBP'
                
            )
            print(f"{lb_out:12.6f} | {ub_out:12.6f}")

    f.close()

        