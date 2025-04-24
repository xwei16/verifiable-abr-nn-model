#!/usr/bin/env python3
# pgd_one_attack.py  ----------------------------------------------
"""
Targeted PGD against Pensieve's actor network - PyTorch implementation
  • pushes every state toward class-0 (300 kbps)
  • perturbs only one feature (Last1_downloadtime) and respects both an L∞ budget (eps)
    and the physical bounds in qoe_spec.json
Outputs a new CSV "pgd_attacked_data.csv" whose feature has been
adversarially modified; all other columns stay identical.
"""

import json
import numpy as np
import pandas as pd
import torch
import random
import collections as C
import os


MODEL_PATH   = "../../pensieve_rl_model/pretrain_linear_reward.pt"
CSV_PATH     = "../../src/filtered_good_testing_data.csv"
SPEC_PATH    = "../../src/qoe_spec.json"
OUT_CSV      = "pgd_attacked_torch_data.csv"

FEATURES = [                         # feature(s) to attack
    "Last1_downloadtime"
]
ATTACK_IDXS = [31]

EPS   = 0.01                        # epsilon for L∞ bound
ALPHA = 0.001                       # step size
STEPS = 40                          # number of PGD steps
TARGET_CLASS = 0                    # target class [0: 300 kbps, 1: 1000 kbps]
M = 0                               # number of spec
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def row_to_state(r, spec):
    """Convert one pandas row to Pensieve state (6×8)."""
    s = np.zeros((6, 8), dtype=np.float32)
    s[0, 0] = r["Last1_chunk_bitrate"]
    s[1, 0] = r["Last1_buffer_size"]
    s[2] = r[[f"Last{i}_throughput"   for i in range(8, 0, -1)]].values
    s[3] = r[[f"Last{i}_downloadtime" for i in range(8, 0, -1)]].values
    s[4, :6] = r[[f"chunksize{i}"        for i in range(1, 7)]].values
    s[5, 0] = r["Chunks_left"]
    #TODO: what is this doing?
    print(len(spec))
    print(spec)
    chosen = random.choice(spec)
    print(len(chosen))
    print(chosen)
    s[3, 7] = random.uniform(
        chosen["Last1_downloadtime_l"],
        chosen["Last1_downloadtime_u"]
    )
    return s


## Load data

df   = pd.read_csv(CSV_PATH)
raw  = json.load(open(SPEC_PATH))
spec  = raw if isinstance(raw, list) else [raw]
M    = len(spec)

low  = np.array([spec[i][f"{k}_l"] for k in FEATURES for i in range(M)],
                dtype=np.float32)
high = np.array([spec[i][f"{k}_u"] for k in FEATURES for i in range(M)],
                dtype=np.float32)

#TODO: something needs to be changed later, ask @xwei16
N = len(df)
states_np = np.stack([row_to_state(r, spec) for _, r in df.iterrows()])  # Shape: (N, 6, 8)
orig_feats = states_np[:, 3, 7].astype(np.float32)


if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    print(f"Loaded native PyTorch model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"PyTorch model not found at {MODEL_PATH}")


## PGD attack

adv_feats = np.empty_like(orig_feats)

model.eval()

for i in range(N):
    state0 = torch.tensor(states_np[i:i+1], device=DEVICE)  # keep batch dim = 1
    x0 = orig_feats[i]  # scalar (one feature)
    
    # Skip if already class-0
    with torch.no_grad():
        pred = torch.argmax(model(state0), dim=1)[0].item()
    
    if pred == TARGET_CLASS:
        adv_feats[i] = x0
        continue
    
    x_adv = x0
    
    for _ in range(STEPS):
        # Create a copy of the state and inject the adversarial value
        state_cur = state0.clone().detach()
        state_cur_flat = state_cur.view(-1)
        state_cur_flat[ATTACK_IDXS[0]] = float(x_adv)
        
        # Set requires_grad
        state_cur.requires_grad = True
        
        # Forward pass
        logits = model(state_cur)
        
        # Compute loss (cross-entropy to target class)
        target = torch.tensor([TARGET_CLASS], device=DEVICE)
        loss = torch.nn.functional.cross_entropy(logits, target)
        
        # Backward pass
        loss.backward()
        
        # Get gradient for the attack index
        grad = state_cur.grad.view(-1)[ATTACK_IDXS[0]].item()
        
        # PGD update
        x_adv = x_adv - ALPHA * np.sign(grad)
        
        # Project back to L∞ ball
        x_adv = np.clip(x_adv, x0 - EPS, x0 + EPS)
        
        # Project to spec bounds
        x_adv = np.clip(x_adv, low, high)
    
    adv_feats[i] = x_adv


## Write to CSV

adv_feats = adv_feats.reshape(-1, 1)  # shape (N, 1)
df.loc[:, FEATURES] = adv_feats
df.to_csv(OUT_CSV, index=False)
print(f"Adversarial CSV written →  {OUT_CSV}")


## Verify attack effectiveness

print("\n▶  Verifying attack effectiveness …")

success = unchanged = higher = 0
orig_preds, adv_preds = [], []

model.eval()
with torch.no_grad():
    for i in range(N):
        state_orig = torch.tensor(states_np[i:i+1], device=DEVICE)
        state_adv = state_orig.clone()
        state_adv_flat = state_adv.view(-1)
        state_adv_flat[ATTACK_IDXS[0]] = adv_feats[i, 0]

        p_orig = torch.argmax(model(state_orig), dim=1)[0].item()
        p_adv = torch.argmax(model(state_adv), dim=1)[0].item()

        orig_preds.append(p_orig)
        adv_preds.append(p_adv)

        if p_adv < p_orig:
            success += 1
        elif p_adv == p_orig:
            unchanged += 1
        else:
            higher += 1

print(f"Total samples           : {N}")
print(f"Successful downgrades   : {success}  ({success / N * 100:.1f}%)")
print(f"Unchanged predictions   : {unchanged}")
print(f"Accidentally higher br  : {higher}")

# Distribution before/after
print("\nBit-rate histogram (index 0-5)")
print("Original    :", dict(sorted(C.Counter(orig_preds).items())))
print("Adversarial :", dict(sorted(C.Counter(adv_preds).items()))) 