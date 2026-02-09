import os
import torch
import torch.nn as nn
import argparse
import subprocess
import torch.nn.functional as F
# python abcrown_pensieve_qoe.py   --model-path ../model/abr-model/pensieve_rl_model/nn_model_ep_155400.pth   --vnnlib-dir spec/  --abcrown alpha-beta-CROWN/complete_verifier/abcrown.py

# -----------------------------
# Pensieve Actor Network
# -----------------------------

class PensieveActor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        # Per-row feature extractors (misnamed "conv" in checkpoint)
        self.conv1_actor = nn.Linear(8, 128)
        self.conv2_actor = nn.Linear(8, 128)
        self.conv3_actor = nn.Linear(6, 128)

        # Fully connected layers
        self.fc1_actor = nn.Linear(1, 128)
        self.fc2_actor = nn.Linear(1, 128)
        self.fc3_actor = nn.Linear(1, 128)
        self.fc4_actor = nn.Linear(128 * 6, 128)

        self.pi_head = nn.Linear(128, a_dim)

    def forward(self, x):
        # x: [B, 6, 8]

        batch = x.size(0)

        # Process first 3 rows with Linear(8 → 128)
        h1 = F.relu(self.conv1_actor(x[:, 0, :]))
        h2 = F.relu(self.conv2_actor(x[:, 1, :]))
        h3 = F.relu(self.conv2_actor(x[:, 2, :]))

        # Third "conv" operates on 6-dim vector (throughput history etc.)
        h4 = F.relu(self.conv3_actor(x[:, :, 0]))

        # Scalar features (buffer, last bitrate, etc.)
        h5 = F.relu(self.fc1_actor(x[:, 3, 0:1]))
        h6 = F.relu(self.fc2_actor(x[:, 4, 0:1]))
        h7 = F.relu(self.fc3_actor(x[:, 5, 0:1]))

        # Concatenate (Pensieve magic number: 6 × 128)
        concat = torch.cat([h1, h2, h3, h4, h5, h6], dim=1)

        hidden = F.relu(self.fc4_actor(concat))
        logits = self.pi_head(hidden)

        # Verification-friendly "softmax"
        probs = F.relu(logits)
        probs = probs / probs.sum(dim=1, keepdim=True)

        return probs
# -----------------------------
# QoE Wrapper (THIS IS CRITICAL)
# -----------------------------
class PensieveQoEWrapper(nn.Module):
    def __init__(self, actor, last_bitrate=1850.0, max_rebuffer=0.2):
        super().__init__()
        self.actor = actor

        # Pensieve bitrate ladder (Kbps)
        bitrates = torch.tensor(
            [300, 750, 1200, 1850, 2850, 4300],
            dtype=torch.float32
        )
        self.register_buffer("bitrates", bitrates)

        self.last_bitrate = last_bitrate
        self.max_rebuffer = max_rebuffer

    def forward(self, x):
        probs = self.actor(x)  # [B, 6]
        expected_bitrate = (probs * self.bitrates).sum(dim=1)

        smoothness = torch.abs(expected_bitrate - self.last_bitrate)

        qoe = (
            expected_bitrate / 1000.0
            - 4.3 * self.max_rebuffer
            - smoothness / 1000.0
        )

        return qoe.unsqueeze(1)  # [B, 1]


# -----------------------------
# Export ONNX
# -----------------------------
def export_onnx(model_path, onnx_path):
    S_DIM = [6, 8]
    A_DIM = 6

    actor = PensieveActor(S_DIM, A_DIM)
    ckpt = torch.load(model_path, map_location="cpu")
    actor.load_state_dict(ckpt[0])
    actor.eval()

    dummy_input = torch.zeros(1, 6, 8)

    torch.onnx.export(
    actor,
    dummy_input,
    onnx_path,
    input_names=["state"],
    output_names=["actions"],
    dynamic_axes={"state": {0: "batch"}},
    opset_version=13
)
    
    print(f"[✓] Exported ONNX to {onnx_path}")


# -----------------------------
# YAML generation (DEFAULT AB-CROWN)
# -----------------------------
def create_yaml(yaml_path, onnx_path, vnnlib_path):
    with open(yaml_path, "w") as f:
        f.write(
            f"""
general:
  complete_verifier: none
  enable_incomplete_verification: true
  save_output: true
  output_file: qoe_bounds.pkl

model:
  onnx_path: onnx_models/pensieve_qoe.onnx
  input_shape: [-1, 6, 8]

specification:
  vnnlib_path: spec/0.vnnlib

attack:
  general_attack: false
  enable_mip_attack: false

bab:
  attack:
    enabled: false
"""
        )


# -----------------------------
# Main runner
# -----------------------------
def main(args):
    os.makedirs(args.onnx_dir, exist_ok=True)
    os.makedirs(args.yaml_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    onnx_path = os.path.join(args.onnx_dir, "pensieve_qoe.onnx")
    export_onnx(args.model_path, onnx_path)

    for fname in sorted(os.listdir(args.vnnlib_dir)):
        if not fname.endswith(".vnnlib"):
            continue

        idx = fname.replace(".vnnlib", "")
        vnn_path = os.path.join(args.vnnlib_dir, fname)
        yaml_path = os.path.join(args.yaml_dir, f"{idx}.yaml")
        out_path = os.path.join(args.result_dir, f"{idx}.txt")

        create_yaml(yaml_path, onnx_path, vnn_path)

        cmd = [
            "python",
            args.abcrown,
            "--config",
            yaml_path
        ]

        print(f"[→] Verifying {fname}")
        with open(out_path, "w") as out:
            subprocess.run(cmd, stdout=out, stderr=out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="nn_model_ep_155400.pth")
    parser.add_argument("--vnnlib-dir", required=True)
    parser.add_argument("--onnx-dir", default="onnx_models")
    parser.add_argument("--yaml-dir", default="yaml")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--abcrown", default="complete_verifier/abcrown.py")
    args = parser.parse_args()

    main(args)