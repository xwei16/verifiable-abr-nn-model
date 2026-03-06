"""
lirpa_pensieve.py
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from bound_splitting import bound_splitting


MAX_ROUND = 2
BRS = [300,750,1200,1850,2850,4300]

# ---------------------------------------------------------------------------
# ENV Network
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Pensieve Actor Network
# ---------------------------------------------------------------------------
class PensieveActor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.conv1_actor = nn.Linear(8, 128)
        self.conv2_actor = nn.Linear(8, 128)
        self.conv3_actor = nn.Linear(6, 128)
        self.fc1_actor = nn.Linear(1, 128)
        self.fc2_actor = nn.Linear(1, 128)
        self.fc3_actor = nn.Linear(1, 128)
        self.fc4_actor = nn.Linear(128 * 6, 128)
        self.pi_head = nn.Linear(128, a_dim)

    def forward(self, x):
        h1 = F.relu(self.conv1_actor(x[:, 0, :]))
        h2 = F.relu(self.conv2_actor(x[:, 1, :]))
        h3 = F.relu(self.conv2_actor(x[:, 2, :]))
        h4 = F.relu(self.conv3_actor(x[:, :, 0]))
        h5 = F.relu(self.fc1_actor(x[:, 3, 0:1]))
        h6 = F.relu(self.fc2_actor(x[:, 4, 0:1]))
        h7 = F.relu(self.fc3_actor(x[:, 5, 0:1]))
        concat = torch.cat([h1, h2, h3, h4, h5, h6], dim=1)
        hidden = F.relu(self.fc4_actor(concat))
        logits = self.pi_head(hidden)
        return logits

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_pensieve_actor(model_path, device):
    actor = PensieveActor([6, 8], 6)
    ckpt = torch.load(model_path, map_location=device)
    actor.load_state_dict(ckpt[0])
    actor.eval()
    actor.to(device)
    return actor

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
def load_initial_state(device):
    lb = np.zeros((6, 8), dtype=np.float32)
    ub = np.zeros((6, 8), dtype=np.float32)

    lb[0, 7] = 0.66279;  ub[0, 7] = 0.66279
    lb[1, 7] = 0.4;      ub[1, 7] = 0.5
    lb[2, 7] = 0.85;     ub[2, 7] = 4.22
    lb[3, 7] = 0.41;     ub[3, 7] = 1.05

    chunk_size_lb = [0.11, 0.26, 0.39, 0.60, 0.89, 1.43]
    chunk_size_ub = [0.18, 0.45, 0.71, 1.08, 1.73, 2.40]
    for i in range(6):
        lb[4, i] = chunk_size_lb[i]
        ub[4, i] = chunk_size_ub[i]

    lb[5, 7] = 0;   ub[5, 7] = 0.96

    if np.any(lb > ub):
        raise ValueError("lb > ub in initial state")

    lb_t = torch.tensor(lb, dtype=torch.float32, device=device).unsqueeze(0)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=device).unsqueeze(0)
    return lb_t, ub_t

# ---------------------------------------------------------------------------
# Pensieve output bounds
# ---------------------------------------------------------------------------
def pensieve_output_bounds(model, lb, ub, method):
    device = lb.device
    dummy = torch.zeros(1, 6, 8, device=device)
    lirpa_model = BoundedModule(model, dummy, device=device)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    x = BoundedTensor((lb + ub) / 2.0, ptb)
    return lirpa_model.compute_bounds(x=(x,), method=method)

# ---------------------------------------------------------------------------
# ENV model input bounds
# ---------------------------------------------------------------------------
def network_prediction_bound(lb_np, ub_np, current_br_idx,
                              past_chunk_size_lb, past_chunk_size_ub,
                              past_download_time_lb, past_download_time_ub):
    lb_list = [0] * 19
    ub_list = [0] * 19

    for i, (l, u) in enumerate(zip(past_chunk_size_lb, past_chunk_size_ub)):
        lb_list[i * 2] = l;  ub_list[i * 2] = u

    for i, (l, u) in enumerate(zip(past_download_time_lb, past_download_time_ub)):
        lb_list[i * 2 + 1] = l * 10;  ub_list[i * 2 + 1] = u * 10

    lb_list[16] = min(lb_np[2, i] for i in range(8))
    ub_list[16] = max(ub_np[2, i] for i in range(8))
    lb_list[17] = 0.01;  ub_list[17] = 0.15
    lb_list[18] = lb_np[4, current_br_idx]
    ub_list[18] = ub_np[4, current_br_idx]

    return np.array(lb_list, dtype=np.float32), np.array(ub_list, dtype=np.float32)

def load_normalization_params(filepath):
    data = np.load(filepath)
    return data['X_max'], data['y_max']

# ---------------------------------------------------------------------------
# ENV model output bounds
# ---------------------------------------------------------------------------
def net_output_bounds(f, model, lb, ub, X_max, y_max, method="IBP"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    X_max = X_max.astype(np.float32)
    y_max = np.float32(y_max)

    f.write(f"│  │  ├─ Input bounds (ENV input, raw):\n"
            f"│  │  │  ├─ Lower: {lb}\n"
            f"│  │  │  └─ Upper: {ub}\n")

    lb = torch.tensor(lb.astype(np.float32) / X_max, dtype=torch.float32).unsqueeze(0).to(device)
    ub = torch.tensor(ub.astype(np.float32) / X_max, dtype=torch.float32).unsqueeze(0).to(device)

    f.write(f"│  │  ├─ Input bounds (ENV input, normalized):\n"
            f"│  │  │  ├─ Lower: {lb}\n"
            f"│  │  │  └─ Upper: {ub}\n")

    dummy = torch.zeros(1, lb.shape[1]).to(device)
    lirpa_model = BoundedModule(model, dummy, device=device)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    x = BoundedTensor((lb + ub) / 2.0, ptb)
    lb_out, ub_out = lirpa_model.compute_bounds(x=(x,), method=method)

    f.write(f"│  │  ├─ Output bounds (normalized):\n"
            f"│  │  │  ├─ Lower: {lb_out}\n"
            f"│  │  │  └─ Upper: {ub_out}\n")

    lb_out = np.float32(max(0.0, lb_out.item() * y_max))
    ub_out = np.float32(ub_out.item() * y_max)

    f.write(f"│  │  └─ Output bounds (denormalized):\n"
            f"│  │     ├─ Lower: {lb_out}\n"
            f"│  │     └─ Upper: {ub_out}\n")

    return lb_out, ub_out

# ---------------------------------------------------------------------------
# Update input bounds
# ---------------------------------------------------------------------------
def update_input_bound(input_lb, input_ub, new_br_idx, new_dt_lb, new_dt_ub, device):
    input_lb[0, 7] = BRS[new_br_idx] / BRS[-1]
    input_ub[0, 7] = BRS[new_br_idx] / BRS[-1]

    input_lb[2] = np.roll(input_lb[2], -1)
    input_ub[2] = np.roll(input_ub[2], -1)
    input_lb[2, -1] = input_lb[4, new_br_idx] / new_dt_ub
    input_ub[2, -1] = 1.5

    input_lb[3] = np.roll(input_lb[3], -1)
    input_ub[3] = np.roll(input_ub[3], -1)
    input_lb[3, 7] = new_dt_lb
    input_ub[3, 7] = new_dt_ub

    if np.any(input_lb > input_ub):
        raise ValueError("lb > ub after update")

    lb_t = torch.tensor(input_lb, dtype=torch.float32, device=device).unsqueeze(0)
    ub_t = torch.tensor(input_ub, dtype=torch.float32, device=device).unsqueeze(0)
    return lb_t, ub_t

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    pensieve_actor = load_pensieve_actor(args.pensieve_model_path, device)
    print(f"[✓] Pensieve Model loaded from {args.pensieve_model_path}")

    env_model = torch.load(args.env_model_dir + "network_pred.pt", weights_only=False)
    print(f"[✓] ENV Model loaded from {args.env_model_dir}")

    X_max, y_max = load_normalization_params(args.env_model_dir + "normalization_params.npz")

    last_br_idx = int(args.last_br_idx)

    import os
    from datetime import datetime
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename       = f"logs/verification_tree_{timestamp}.log"
    jsonl_log_filename = f"logs/verification_tree_{timestamp}.jsonl"

    # Single shared node counter across ALL rounds and ALL bab_search calls
    # so every node in the entire run has a globally unique ID.
    node_counter = [0]

    with open(log_filename, "w") as f, open(jsonl_log_filename, "w") as jsonl_f:

        f.write("="*70 + "\n")
        f.write("VERIFICATION LOOP WITH BOUND SPLITTING\n")
        f.write("="*70 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"Pensieve Model: {args.pensieve_model_path}\n")
        f.write(f"ENV Model: {args.env_model_dir}\n")
        f.write(f"Method: CROWN-Optimized\n")
        f.write(f"Initial BR Index: {last_br_idx}\n")
        f.write("="*70 + "\n\n")
        f.flush()

        method = "CROWN-Optimized"
        input_lb, input_ub = load_initial_state(device)

        past_chunk_size_lb    = np.zeros(8, dtype=np.float32)
        past_chunk_size_ub    = np.zeros(8, dtype=np.float32)
        past_download_time_lb = np.zeros(8, dtype=np.float32)
        past_download_time_ub = np.zeros(8, dtype=np.float32)

        # Each entry: (input_lb, input_ub, chunk_size_lb, chunk_size_ub,
        #              download_time_lb, download_time_ub, br_idx, parent_node_id)
        # parent_node_id is the safe-region node from the previous round that
        # produced this input — None for the very first call.
        current_regions = [(
            input_lb, input_ub,
            past_chunk_size_lb.copy(), past_chunk_size_ub.copy(),
            past_download_time_lb.copy(), past_download_time_ub.copy(),
            last_br_idx,
            None,   # ← parent_node_id: None for round 0
        )]

        for round_num in range(MAX_ROUND):
            f.write(f"\n{'='*70}\n")
            f.write(f"ROUND {round_num}\n")
            f.write(f"Regions to process: {len(current_regions)}\n")
            f.write(f"{'='*70}\n\n")

            next_regions = []

            for region_idx, (input_lb, input_ub,
                             chunk_size_lb, chunk_size_ub,
                             download_time_lb, download_time_ub,
                             current_br_idx, parent_node_id) in enumerate(current_regions):

                f.write(f"\n[Region {region_idx + 1}/{len(current_regions)}]\n")
                f.write(f"{'─'*70}\n")

                input_lb_np = input_lb.squeeze(0).cpu().numpy()
                input_ub_np = input_ub.squeeze(0).cpu().numpy()

                f.write(f"├─ Input Bounds (Pensieve Input):\n"
                        f"│  ├─ Lower: {input_lb_np}\n"
                        f"│  ├─ Upper: {input_ub_np}\n"
                        f"│  └─ Last BR Index: {current_br_idx}\n")

                logit_lb, logit_ub = pensieve_output_bounds(
                    pensieve_actor, input_lb, input_ub, method)
                f.write(f"│\n├─ Logit Bounds (Pensieve Output):\n"
                        f"│  ├─ Lower: {logit_lb.detach().cpu().numpy()}\n"
                        f"│  └─ Upper: {logit_ub.detach().cpu().numpy()}\n")

                f.write(f"│\n├─ BOUND SPLITTING SEARCH:\n│  │\n")

                # Pass the shared counter and the parent link so the new root
                # is connected to parent_node_id in the global tree.
                safe_regions = bound_splitting(
                    pensieve_actor, input_lb, input_ub,
                    node_counter=node_counter,
                    parent_node_id=parent_node_id,
                    log_file=f,
                    jsonl_file=jsonl_f,
                    level=round_num,
                )

                f.write(f"│\n├─ Bound Splitting Results:\n"
                        f"│  ├─ Safe Regions Found: {len(safe_regions)}\n")
                for j, (lb_r, ub_r, action, nid) in enumerate(safe_regions):
                    f.write(f"│  ├─ Region {j+1} [Node {nid}]: "
                            f"Throughput {lb_r[0,2,7].item():.6f}"
                            f" ~ {ub_r[0,2,7].item():.6f}"
                            f" -> Action {action}\n")

                # Process each safe region through ENV model
                for safe_idx, (lb_r, ub_r, br_idx, safe_node_id) in enumerate(safe_regions):
                    f.write(f"│\n├─ Processing Safe Region {safe_idx + 1} [Node {safe_node_id}]:\n")

                    input_lb_r_np = lb_r.squeeze(0).cpu().numpy()
                    input_ub_r_np = ub_r.squeeze(0).cpu().numpy()

                    env_lb, env_ub = network_prediction_bound(
                        input_lb_r_np, input_ub_r_np, br_idx,
                        chunk_size_lb, chunk_size_ub,
                        download_time_lb, download_time_ub,
                    )
                    f.write(f"│  │\n")
                    dt_lb, dt_ub = net_output_bounds(
                        f, env_model, env_lb, env_ub, X_max, y_max, method)

                    new_past_chunk_size_lb = np.roll(chunk_size_lb, -1)
                    new_past_chunk_size_lb[-1] = np.float32(input_lb_r_np[4, 2])
                    new_past_chunk_size_ub = np.roll(chunk_size_ub, -1)
                    new_past_chunk_size_ub[-1] = np.float32(input_ub_r_np[4, 2])
                    new_past_download_time_lb = np.roll(download_time_lb, -1)
                    new_past_download_time_lb[-1] = np.float32(dt_lb)
                    new_past_download_time_ub = np.roll(download_time_ub, -1)
                    new_past_download_time_ub[-1] = np.float32(dt_ub)

                    next_lb, next_ub = update_input_bound(
                        input_lb_r_np, input_ub_r_np, br_idx, dt_lb, dt_ub, device)

                    if round_num < MAX_ROUND - 1:
                        next_regions.append((
                            next_lb, next_ub,
                            new_past_chunk_size_lb, new_past_chunk_size_ub,
                            new_past_download_time_lb, new_past_download_time_ub,
                            br_idx,
                            safe_node_id,  # ← this safe region becomes the parent
                        ))
                        f.write(f"│  └─ Stored for next round (parent node {safe_node_id})\n")
                    else:
                        f.write(f"│  └─ Final round, not queued\n")

            current_regions = next_regions

            f.write(f"\n{'='*70}\n"
                    f"Round {round_num} complete: {len(next_regions)} regions for next round\n"
                    f"{'='*70}\n")
            f.flush()

            if not current_regions:
                f.write("\nNo regions for next round, stopping.\n")
                break

        f.write(f"\n{'='*70}\n"
                f"HIERARCHICAL VERIFICATION COMPLETE\n"
                f"Total nodes assigned: {node_counter[0]}\n"
                f"{'='*70}\n")
        f.flush()

    print(f"\n[✓] Text log  : {log_filename}")
    print(f"[✓] JSONL log : {jsonl_log_filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pensieve-model-path", required=True)
    parser.add_argument("--last-br-idx", required=True)
    parser.add_argument("--env-model-dir", required=True)
    args = parser.parse_args()
    main(args)