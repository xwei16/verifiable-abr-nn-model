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
# python3 lirpa_pensieve.py --pensieve-model-path \
# ../model/abr-model/pensieve_rl_model/nn_model_ep_155400.pth \
# --env-model-dir ../model/network-prediction-model/
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

past_chunk_size_lb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
past_chunk_size_ub = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
past_download_time_lb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
past_download_time_ub = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

MAX_ROUND = 1

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
        # probs = F.relu(logits)
        # probs = probs / probs.sum(dim=1, keepdim=True)
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
def load_bounds_from_spec(spec: dict, device: torch.device):
    """
    Convert one entry from the ABR JSON spec into lower/upper bound tensors
    shaped (1, 6, 8) matching Pensieve's input layout.

    Only the input bound fields are used:
      Last1_chunk_bitrate, Last1_buffer_size, Last1-8_throughput,
      Last1-8_downloadtime, chunksize1-6, Chunks_left.

    cur_br and qoe_2 are intentionally ignored — they are spec metadata,
    not network inputs.

    All positions that Pensieve never writes remain 0 in both lb and ub.

    Parameters
    ----------
    spec : dict
        One spec entry (already parsed from JSON).
    device : torch.device

    Returns
    -------
    lb : torch.Tensor  shape (1, 6, 8)
    ub : torch.Tensor  shape (1, 6, 8)
    """
    lb = np.zeros((6, 8), dtype=np.float32)
    ub = np.zeros((6, 8), dtype=np.float32)

    # ------------------------------------------------------------------
    # Row 2: throughput history — Last8 (oldest) → col 0, Last1 → col 7
    # ------------------------------------------------------------------
    for n in range(1, 9):          # n = 1 .. 8
        col = 8 - n                # Last1 → col 7, Last8 → col 0
        lb[2, col] = spec[f"Last{n}_throughput_l"]
        ub[2, col] = spec[f"Last{n}_throughput_u"]

    # ------------------------------------------------------------------
    # Row 3: download time history — same convention
    # ------------------------------------------------------------------
    for n in range(1, 9):
        col = 8 - n
        lb[3, col] = spec[f"Last{n}_downloadtime_l"]
        ub[3, col] = spec[f"Last{n}_downloadtime_u"]

    # ------------------------------------------------------------------
    # Row 4: next chunk sizes for 6 quality levels — cols 0..5
    # ------------------------------------------------------------------
    for i in range(1, 7):         # chunksize1 .. chunksize6
        lb[4, i - 1] = spec[f"chunksize{i}_l"]
        ub[4, i - 1] = spec[f"chunksize{i}_u"]
    # cols 6 and 7 of row 4 remain 0 (Pensieve never writes them)

    # ------------------------------------------------------------------
    # Row 0: last chunk bitrate — only col 7 (s[0, -1])
    # ------------------------------------------------------------------
    lb[0, 7] = spec["Last1_chunk_bitrate_l"]
    ub[0, 7] = spec["Last1_chunk_bitrate_u"]

    # ------------------------------------------------------------------
    # Row 1: buffer size — only col 7 (s[1, -1])
    # ------------------------------------------------------------------
    lb[1, 7] = spec["Last1_buffer_size_l"]
    ub[1, 7] = spec["Last1_buffer_size_u"]

    # ------------------------------------------------------------------
    # Row 5: chunks left — only col 7 (s[5, -1])
    # ------------------------------------------------------------------
    lb[5, 7] = spec["Chunks_left_l"]
    ub[5, 7] = spec["Chunks_left_u"]

    # Sanity check
    if np.any(lb > ub):
        raise ValueError("Spec contains lb > ub for at least one input dimension.")

    lb_t = torch.tensor(lb, dtype=torch.float32, device=device).unsqueeze(0)  # (1,6,8)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=device).unsqueeze(0)
    return lb_t, ub_t


# ---------------------------------------------------------------------------
# auto-LiRPA bound computation
# ---------------------------------------------------------------------------
def compute_output_bounds(
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
# Pretty-print helper
# ---------------------------------------------------------------------------
def _fmt(arr: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in arr.flatten()) + "]"


def network_prediction_bound(lb_np, ub_np):
    """
    lb_np, ub_np shape: (6, 8)
    Pensieve layout:
        s[0] = last bitrate (_ / 4300)
        s[1] = buffer (sec / 10)
        s[2] = throughput (MB/s)
        s[3] = download time (sec / 10)
        s[4] = chunk sizes (MB) - next chunk only
        s[5] = chunks left (_ / 48)
    """

    env_input = {}

    # Chunk sizes (MB)
    chunk_bounds = []
    for i in range(8):
        chunk_bounds.append((
            lb_np[4, i],
            ub_np[4, i]
        ))
    env_input["chunk_size_MB"] = chunk_bounds

    # Download time (seconds)
    dt_bounds = []
    for i in range(8):
        dt_bounds.append((
            lb_np[3, i] * 10.0,
            ub_np[3, i] * 10.0
        ))
    env_input["download_time_sec"] = dt_bounds

    # Bandwidth (MBps)
    bw_l_all = []
    bw_u_all = []

    for i in range(8):
        bw_l_all.append(lb_np[2, i])
        bw_u_all.append(ub_np[2, i])

    B_min = min(bw_l_all)
    B_max = max(bw_u_all)

    env_input["next_bandwidth_MBps"] = (B_min, B_max)

    # Next network delay (propagation-only approx)
    # delay ≈ download_time - (chunk_size / bandwidth)
    network_delay_intervals = []
    bw_bounds = []
    for i in range(8):
        bw_bounds.append((
            lb_np[2, i],
            ub_np[2, i]
        ))

    for i in range(8):

        size_l, size_u = chunk_bounds[i]
        dt_l, dt_u = dt_bounds[i]
        bw_l_MBps, bw_u_MBps = bw_bounds[i]

        if bw_l_MBps <= 1e-9:
            continue

        tx_min = size_l / bw_u_MBps
        tx_max = size_u / bw_l_MBps

        nd_l = max(0.0, dt_l - tx_max)
        nd_u = max(0.0, dt_u - tx_min)

        network_delay_intervals.append((nd_l, nd_u))

    if network_delay_intervals:
        env_input["next_network_delay_sec"] = (
            min(v[0] for v in network_delay_intervals),
            max(v[1] for v in network_delay_intervals),
        )
    else:
        env_input["next_network_delay_sec"] = (0.0, 0.0)

    return env_input

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Load model
    pensieve_actor = load_pensieve_actor(args.pensieve_model_path, device)
    print(f"[✓] Pensieve Model loaded from {args.pensieve_model_path}")

    env_model = torch.load(args.env_model_dir + "network_pred.pt", weights_only=False)
    env_norm_param_file = args.env_model_dir + "normalization_params.npz"
    print(f"[✓] ENV Model loaded from {args.env_model_dir}")

    # Load spec file — run every entry - start state (S0)
    with open(args.spec_path) as f:
        all_specs = json.load(f)

    # open log file
    with open("logs/lirpa.log", "w") as f:
        specs_to_run = list(enumerate(all_specs))

        print(f"[✓] Spec file loaded: {len(all_specs)} entries, running all")

        methods = ["IBP", "CROWN", "CROWN-Optimized"]

        for spec_idx, spec in specs_to_run:
            f.write(f"{'='*70}")
            f.write(f"Spec [{spec_idx}]")
            f.write(f"{'='*70}\n")

            lb, ub = load_bounds_from_spec(spec, device)

            # Summarise the non-trivial input bounds
            lb_np = lb.squeeze(0).cpu().numpy()   # (6, 8)
            ub_np = ub.squeeze(0).cpu().numpy()

            f.write(
                f"input bounds:\n"
                f"lower: {lb_np}\n"
                f"upper: {ub_np}\n"
            )

            # print(f"  Bitrate (s[0,7]):       [{lb_np[0,7]:.5f}, {ub_np[0,7]:.5f}]")
            # print(f"  Buffer  (s[1,7]):       [{lb_np[1,7]:.5f}, {ub_np[1,7]:.5f}]")
            # print(f"  Throughput (s[2,0:8]):  lb={lb_np[2]}")
            # print(f"                          ub={ub_np[2]}")
            # print(f"  DL time  (s[3,0:8]):    lb={lb_np[3]}")
            # print(f"                          ub={ub_np[3]}")
            # print(f"  Chunk sz (s[4,0:6]):    lb={lb_np[4,:6]}")
            # print(f"                          ub={ub_np[4,:6]}")
            # print(f"  Chunks left (s[5,7]):   [{lb_np[5,7]:.5f}, {ub_np[5,7]:.5f}]")

            for method in methods:

                f.write(f"--- {method} ---\n")
                for _ in range(MAX_ROUND):
                    # pensieve logit bound 
                    lb_out, ub_out = compute_output_bounds(pensieve_actor, lb, ub, method)
                    lb_arr = lb_out.detach().cpu().numpy()
                    ub_arr = ub_out.detach().cpu().numpy()
                    print(f"  lb: {_fmt(lb_arr)}")
                    print(f"  ub: {_fmt(ub_arr)}")
                    f.write(
                        f"logit bounds:\n"
                        f"lower: {lb_arr}\n"
                        f"upper: {ub_arr}\n"
                    )

                    # ENV bound
                    # env_inputs = network_prediction_bound(lb_np, ub_np) # TODO

                    # print("  Bandwidth (Mbps):")
                    # for i, (l, u) in enumerate(env_inputs["bandwidth_Mbps"], 1):
                    #     print(f"    t-{i}: [{l:.3f}, {u:.3f}]")

                    # print("\n  Download Time (sec):")
                    # for i, (l, u) in enumerate(env_inputs["download_time_sec"], 1):
                    #     print(f"    t-{i}: [{l:.3f}, {u:.3f}]")

                    # print("\n  Chunk Size (MB):")
                    # for i, (l, u) in enumerate(env_inputs["chunk_size_MB"], 1):
                    #     print(f"    t-{i}: [{l:.3f}, {u:.3f}]")

                    # nb_l, nb_u = env_inputs["next_bandwidth_Mbps"]
                    # print(f"\n  Next Bandwidth Envelope (Mbps): [{nb_l:.3f}, {nb_u:.3f}]")

                    # nd_l, nd_u = env_inputs["next_network_delay_sec"]
                    # print(f"  Next Network Delay Approx (sec): [{nd_l:.3f}, {nd_u:.3f}]")


            # TODO: compute delta QoE bound

            # Save per-spec results
            # out_path = f"{args.output_prefix}_spec{spec_idx:04d}.npz"
            # save_dict = {
            #     "spec_index":  spec_idx,
            #     "lb_input":    lb_np,
            #     "ub_input":    ub_np,
            # }
            # for method, res in results.items():
            #     key = method.replace("-", "_")
            #     save_dict[f"{key}_lower"] = res["lower"]
            #     save_dict[f"{key}_upper"] = res["upper"]

            # np.savez(out_path, **save_dict)
            # print(f"\n  [✓] Saved → {out_path}")

            

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run auto-LiRPA verification on Pensieve actor using ABR JSON spec bounds."
    )
    parser.add_argument(
        "--pensieve-model-path",
        required=True,
        help="Path to the Pensieve checkpoint (.pt)",
    )
    parser.add_argument(
        "--env-model-dir",
        required=True,
        help="Path to the directory of the ENV checkpoint",
    )
    parser.add_argument(
        "--spec-path",
        default="../data/abr-specifications/full_spec_no_300.json",
        help="Path to the ABR JSON spec file",
    )
    parser.add_argument(
        "--output-prefix",
        default="pensieve_bounds",
        help="Prefix for output .npz files",
    )
    args = parser.parse_args()
    main(args)