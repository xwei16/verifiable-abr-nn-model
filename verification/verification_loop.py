"""
lirpa_pensieve.py

Loads input bounds directly from the ABR JSON spec file and runs
auto-LiRPA (IBP / CROWN / CROWN-Optimized) on the Pensieve actor.

Pensieve input tensor layout:  s = np.zeros((6, 8), dtype=np.float32)

  Row 0  s[0, :]     last chunk bitrate history  — spec: Last1_chunk_bitrate (only slot 7 used)
  Row 1  s[1, :]     buffer size history          — spec: Last1_buffer_size   (only slot 7 used)
  Row 2  s[2, 0:8]   throughput history           — spec: Last8..Last1_throughput  (all 8 slots)
  Row 3  s[3, 0:8]   download time history        — spec: Last8..Last1_downloadtime (all 8 slots)
  Row 4  s[4, 0:6]   next chunk sizes             — spec: chunksize1..6
  Row 5  s[5, :]     chunks left                  — spec: Chunks_left         (only slot 7 used)

Indexing convention for throughput / downloadtime history:
  LastN  →  slot (8 - N),  i.e. Last8 (oldest) → index 0, Last1 (newest) → index 7
"""
# python3 verification_loop.py --pensieve-model-path \
# ../model/abr-model/pensieve_rl_model/nn_model_ep_155400.pth \
# --last-br-idx 4 \
# --env-model-dir ../model/network-prediction-model/
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
        h1 = F.relu(self.conv1_actor(x[:, 0, :]))    # bitrate history row
        h2 = F.relu(self.conv2_actor(x[:, 1, :]))    # buffer history row
        h3 = F.relu(self.conv2_actor(x[:, 2, :]))    # throughput history row
        h4 = F.relu(self.conv3_actor(x[:, :, 0]))    # first column across rows

        h5 = F.relu(self.fc1_actor(x[:, 3, 0:1]))   # download time (scalar)
        h6 = F.relu(self.fc2_actor(x[:, 4, 0:1]))   # chunk sizes  (scalar)
        h7 = F.relu(self.fc3_actor(x[:, 5, 0:1]))   # chunks left  (scalar)

        concat = torch.cat([h1, h2, h3, h4, h5, h6], dim=1)
        hidden = F.relu(self.fc4_actor(concat))
        logits = self.pi_head(hidden)
        return logits


# ---------------------------------------------------------------------------
# Pensieve Model loading
# ---------------------------------------------------------------------------
def load_pensieve_actor(model_path: str, device: torch.device) -> PensieveActor:
    actor = PensieveActor([6, 8], 6)
    ckpt = torch.load(model_path, map_location=device)
    actor.load_state_dict(ckpt[0])
    actor.eval()
    actor.to(device)
    return actor

# ---------------------------------------------------------------------------
# JSON spec → (lb, ub) tensors of shape (1, 6, 8)
# ---------------------------------------------------------------------------
def load_initial_state(device: torch.device):
    lb = np.zeros((6, 8), dtype=np.float32)
    ub = np.zeros((6, 8), dtype=np.float32)

    # Row 0: last chunk bitrate (col 7)
    lb[0, 7] = 0.66279
    ub[0, 7] = 0.66279

    # Row 1: buffer size (col 7)
    lb[1, 7] = 0.4
    ub[1, 7] = 0.5 

    # Row 2: throughput history — Last8 (col 0) → Last1 (col 7)
    lb[2, 7] = 0.85
    ub[2, 7] = 4.22

    # Row 3: download time history
    lb[3, 7] = 0.41
    ub[3, 7] = 1.05

    # Row 4: chunk sizes (cols 0..5)
    chunk_size_lb = [0.11, 0.26, 0.39, 0.60, 0.89, 1.43]
    chunk_size_ub = [0.18, 0.45, 0.71, 1.08, 1.73, 2.40]
    for i in range(6):
        lb[4, i] = chunk_size_lb[i]
        ub[4, i] = chunk_size_ub[i]
    
    # Row 5: chunks left (col 7)
    lb[5, 7] = 0
    ub[5, 7] = 0.96

    if np.any(lb > ub):
        raise ValueError("Spec contains lb > ub for at least one input dimension.")

    lb_t = torch.tensor(lb, dtype=torch.float32, device=device).unsqueeze(0)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=device).unsqueeze(0)
    return lb_t, ub_t


# ---------------------------------------------------------------------------
# Pensieve Output Bound Generation
# ---------------------------------------------------------------------------
def pensieve_output_bounds(
    model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    method: str,
):
    device = lb.device
    dummy  = torch.zeros(1, 6, 8, device=device)

    lirpa_model = BoundedModule(model, dummy, device=device)

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    center = (lb + ub) / 2.0
    x = BoundedTensor(center, ptb)

    lb_out, ub_out = lirpa_model.compute_bounds(x=(x,), method=method)
    return lb_out, ub_out

# ---------------------------------------------------------------------------
# ENV Model Input Bound Generation
# ---------------------------------------------------------------------------
def network_prediction_bound(
    lb_np,
    ub_np,
    current_br_idx,
    past_chunk_size_lb,
    past_chunk_size_ub,
    past_download_time_lb,
    past_download_time_ub,
):
    lb_list = [0] * 19
    ub_list = [0] * 19

    for i, (l, u) in enumerate(zip(past_chunk_size_lb, past_chunk_size_ub)):
        lb_list[i * 2] = l
        ub_list[i * 2] = u

    for i, (l, u) in enumerate(zip(past_download_time_lb, past_download_time_ub)):
        lb_list[i * 2 + 1] = l * 10
        ub_list[i * 2 + 1] = u * 10

    bw_l_all = [lb_np[2, i] for i in range(8)]
    bw_u_all = [ub_np[2, i] for i in range(8)]
    lb_list[16] = min(bw_l_all)
    ub_list[16] = max(bw_u_all)

    lb_list[17] = 0.01
    ub_list[17] = 0.15

    lb_list[18] = lb_np[4, current_br_idx]
    ub_list[18] = ub_np[4, current_br_idx]

    return np.array(lb_list, dtype=np.float32), np.array(ub_list, dtype=np.float32)


def load_normalization_params(filepath):
    data = np.load(filepath)
    return data['X_max'], data['y_max']


# ---------------------------------------------------------------------------
# Compute ENV model output bound
# ---------------------------------------------------------------------------
def net_output_bounds(
    f,
    model: nn.Module,
    lb,
    ub,
    X_max, 
    y_max,
    method: str = "IBP",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    X_max = X_max.astype(np.float32)
    y_max = np.float32(y_max)

    f.write(
        f"│  │  ├─ Input bounds (ENV input, raw):\n"
        f"│  │  │  ├─ Lower: {lb}\n"
        f"│  │  │  └─ Upper: {ub}\n"
    )

    lb = lb.astype(np.float32) / X_max
    ub = ub.astype(np.float32) / X_max

    lb = torch.tensor(lb, dtype=torch.float32).unsqueeze(0).to(device)
    ub = torch.tensor(ub, dtype=torch.float32).unsqueeze(0).to(device)
    
    f.write(
        f"│  │  ├─ Input bounds (ENV input, normalized):\n"
        f"│  │  │  ├─ Lower: {lb}\n"
        f"│  │  │  └─ Upper: {ub}\n"
    )

    dummy = torch.zeros(1, lb.shape[1]).to(device)
    lirpa_model = BoundedModule(model, dummy, device=device)

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    center = (lb + ub) / 2.0
    x = BoundedTensor(center, ptb)

    lb_out, ub_out = lirpa_model.compute_bounds(x=(x,), method=method)
    f.write(
        f"│  │  ├─ Output bounds (normalized):\n"
        f"│  │  │  ├─ Lower: {lb_out}\n"
        f"│  │  │  └─ Upper: {ub_out}\n"
    )
    lb_out = np.float32(max(0.0, lb_out.item() * y_max))
    ub_out = np.float32(ub_out.item() * y_max)
    
    f.write(
        f"│  │  └─ Output bounds (denormalized):\n"
        f"│  │     ├─ Lower: {lb_out}\n"
        f"│  │     └─ Upper: {ub_out}\n"
    )

    return lb_out, ub_out


# ---------------------------------------------------------------------------
# Compute delta QOE
# ---------------------------------------------------------------------------
def get_delta_qoe(br_idx, last_br_idx):
    last_q = np.log(BRS[last_br_idx] / BRS[0])
    cur_q = np.log(BRS[br_idx] / BRS[0])
    delta_qoe = cur_q - abs(cur_q - last_q)
    return round(delta_qoe, 5)


# ---------------------------------------------------------------------------
# Update Pensieve Input Bounds
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
        raise ValueError("Spec contains lb > ub for at least one input dimension.")

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

    env_norm_param_file = args.env_model_dir + "normalization_params.npz"
    X_max, y_max = load_normalization_params(env_norm_param_file)
    
    last_br_idx = int(args.last_br_idx)

    import os
    from datetime import datetime
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename      = f"logs/verification_tree_{timestamp}.log"
    jsonl_log_filename = f"logs/verification_tree_{timestamp}.jsonl"
    
    # Open both log files once — JSONL file stays open for incremental writes
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

        past_chunk_size_lb    = np.zeros((8), dtype=np.float32)
        past_chunk_size_ub    = np.zeros((8), dtype=np.float32)
        past_download_time_lb = np.zeros((8), dtype=np.float32)
        past_download_time_ub = np.zeros((8), dtype=np.float32)

        current_regions = [(input_lb, input_ub, past_chunk_size_lb.copy(), past_chunk_size_ub.copy(),
                            past_download_time_lb.copy(), past_download_time_ub.copy(), last_br_idx)]

        for round_num in range(MAX_ROUND):
            f.write(f"\n{'='*70}\n")
            f.write(f"ROUND {round_num}\n")
            f.write(f"Regions to process: {len(current_regions)}\n")
            f.write(f"{'='*70}\n\n")

            next_regions = []
            
            for region_idx, (input_lb, input_ub, chunk_size_lb, chunk_size_ub, 
                           download_time_lb, download_time_ub, current_br_idx) in enumerate(current_regions):
                
                f.write(f"\n[Region {region_idx + 1}/{len(current_regions)}]\n")
                f.write(f"{'─'*70}\n")

                input_lb_np = input_lb.squeeze(0).cpu().numpy()
                input_ub_np = input_ub.squeeze(0).cpu().numpy()

                f.write(
                    f"├─ Input Bounds (Pensieve Input):\n"
                    f"│  ├─ Lower: {input_lb_np}\n"
                    f"│  ├─ Upper: {input_ub_np}\n"
                    f"│  └─ Last BR Index: {current_br_idx}\n"
                )

                logit_lb, logit_ub = pensieve_output_bounds(pensieve_actor, input_lb, input_ub, method)
                logit_lb_np = logit_lb.detach().cpu().numpy()
                logit_ub_np = logit_ub.detach().cpu().numpy()

                f.write(
                    f"│\n├─ Logit Bounds (Pensieve Output):\n"
                    f"│  ├─ Lower: {logit_lb_np}\n"
                    f"│  └─ Upper: {logit_ub_np}\n"
                )

                f.write(f"│\n├─ BOUND SPLITTING SEARCH:\n")
                f.write(f"│  │\n")
                
                # Pass jsonl_f directly — nodes are written as they are certified
                safe_regions = bound_splitting(
                    pensieve_actor, input_lb, input_ub,
                    log_file=f,
                    jsonl_file=jsonl_f,
                    level=round_num,
                )
                
                f.write(f"│\n├─ Bound Splitting Results:\n")
                f.write(f"│  ├─ Safe Regions Found: {len(safe_regions)}\n")
                for j, (lb_r, ub_r, action) in enumerate(safe_regions):
                    f.write(f"│  ├─ Region {j+1}: Throughput {lb_r[0,2,7].item():.6f} ~ {ub_r[0,2,7].item():.6f} -> Action {action}\n")

                for safe_idx, (lb_r, ub_r, br_idx) in enumerate(safe_regions):
                    f.write(f"│\n├─ Processing Safe Region {safe_idx + 1}:\n")
                    
                    input_lb_r_np = lb_r.squeeze(0).cpu().numpy()
                    input_ub_r_np = ub_r.squeeze(0).cpu().numpy()

                    env_lb, env_ub = network_prediction_bound(
                        input_lb_r_np,
                        input_ub_r_np,
                        br_idx,
                        chunk_size_lb,
                        chunk_size_ub,
                        download_time_lb,
                        download_time_ub,
                    )
                    f.write(f"│  │\n")
                    dt_lb, dt_ub = net_output_bounds(f, env_model, env_lb, env_ub, X_max, y_max, method)

                    new_past_chunk_size_lb = np.roll(chunk_size_lb, -1)
                    new_past_chunk_size_lb[-1] = np.float32(input_lb_r_np[4, 2])
                    new_past_chunk_size_ub = np.roll(chunk_size_ub, -1)
                    new_past_chunk_size_ub[-1] = np.float32(input_ub_r_np[4, 2])
                    new_past_download_time_lb = np.roll(download_time_lb, -1)
                    new_past_download_time_lb[-1] = np.float32(dt_lb)
                    new_past_download_time_ub = np.roll(download_time_ub, -1)
                    new_past_download_time_ub[-1] = np.float32(dt_ub)

                    next_lb, next_ub = update_input_bound(input_lb_r_np, input_ub_r_np, br_idx, dt_lb, dt_ub, device)
                    
                    if round_num < MAX_ROUND - 1:
                        next_regions.append((next_lb, next_ub, new_past_chunk_size_lb, new_past_chunk_size_ub,
                                           new_past_download_time_lb, new_past_download_time_ub, br_idx))
                        f.write(f"│  └─ Stored for next round\n")
                    else:
                        f.write(f"│  └─ Final round, not queued\n")

            current_regions = next_regions
            
            f.write(f"\n{'='*70}\n")
            f.write(f"Round {round_num} complete: {len(next_regions)} regions for next round\n")
            f.write(f"{'='*70}\n")
            f.flush()
            
            if not current_regions:
                f.write(f"\nNo regions for next round, stopping exploration.\n")
                break

        f.write(f"\n{'='*70}\n")
        f.write(f"HIERARCHICAL VERIFICATION COMPLETE\n")
        f.write(f"{'='*70}\n")
        f.flush()

    print(f"\n[✓] Text log  : {log_filename}")
    print(f"[✓] JSONL log : {jsonl_log_filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run auto-LiRPA verification on Pensieve actor using ABR JSON spec bounds."
    )
    parser.add_argument("--pensieve-model-path", required=True,
                        help="Path to the Pensieve checkpoint (.pt)")
    parser.add_argument("--last-br-idx", required=True,
                        help="initial br index")
    parser.add_argument("--env-model-dir", required=True,
                        help="Path to the directory of the ENV checkpoint")
    args = parser.parse_args()
    main(args)