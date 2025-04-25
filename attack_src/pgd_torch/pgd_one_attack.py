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

from tqdm import tqdm

import ppo2 as network


MODEL_PATH   = "../../pensieve_rl_model/nn_model_ep_155400.pth"
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
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO2 parameters
S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4


def row_to_state(r, spec):
    """Convert one pandas row to Pensieve state (6×8)."""
    s = np.zeros((6, 8), dtype=np.float32)
    s[0, 0] = r["Last1_chunk_bitrate"]
    s[1, 0] = r["Last1_buffer_size"]
    s[2] = r[[f"Last{i}_throughput"   for i in range(8, 0, -1)]].values
    s[3] = r[[f"Last{i}_downloadtime" for i in range(8, 0, -1)]].values
    s[4, :6] = r[[f"chunksize{i}"        for i in range(1, 7)]].values
    s[5, 0] = r["Chunks_left"]
    # print(len(spec))
    # print(spec)
    chosen = random.choice(spec)
    # print(len(chosen))
    # print(chosen)
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

#TODO: [3, 7] is the index of Last1_downloadtime
N = len(df)
states_np = np.stack([row_to_state(r, spec) for _, r in df.iterrows()])  # Shape: (N, 6, 8)
orig_feats = states_np[:, 3, 7].astype(np.float32)

if os.path.exists(MODEL_PATH):
    # TODO: load parameters from file
    actor = network.Network(state_dim=S_DIM, 
                            action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)
    model = actor.load_model(MODEL_PATH)
    model = actor
    # TODO: move to GPU
    # model.to(DEVICE)
    print(model)
else:
    raise FileNotFoundError(f"PyTorch model not found at {MODEL_PATH}")


## PGD attack

adv_feats = np.empty_like(orig_feats)

# model.eval()


for i in tqdm(range(N)):

    # state0 = torch.tensor(states_np[i:i+1], device=DEVICE).cpu().detach().numpy()
    state0 = states_np[i:i+1]
    x0 = orig_feats[i]
    
    pred = model.predict(state0)
    print(f"pred: {pred}")
    pred = np.argmax(pred)

    
    if pred == TARGET_CLASS:
        adv_feats[i] = x0
        continue
    
    x_adv = x0
    
    for _ in range(STEPS):
        state_cur = state0.clone().detach().requires_grad_(True).flatten()
        state_cur_flat = state_cur.view(-1)
        state_cur_flat[ATTACK_IDXS[0]] = float(x_adv)
        
        output = np.argmax(model.predict(state_cur))
        target = torch.tensor([TARGET_CLASS], device=DEVICE)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        grad = state_cur.grad.view(-1)[ATTACK_IDXS[0]].item()

        x_perturbation = ALPHA * torch.sign(grad)
        x_adv += x_perturbation
        x_adv = torch.clamp(x_adv, x0 - EPS, x0 + EPS)
        x_adv = torch.clamp(x_adv, low, high)

        if i == 0:
            print(f"x_adv: {x_adv}")
            print(f"x0: {x0}")
            print(f"x_perturbation: {x_perturbation}")
            print(f"grad: {grad}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"loss: {loss}")
    
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

with torch.no_grad():
    for i in range(N):
        # state_orig = states_np[i:i+1]
        # state_adv = state_orig.clone().detach().requires_grad_(True).flatten()
        # state_adv_flat = state_adv.view(-1)
        # state_adv_flat[ATTACK_IDXS[0]] = float(x_adv)

        state_orig = states_np[i:i+1]
        state_orig_flat = state_orig.flatten()
        state_adv = state_orig_flat.copy()
        state_adv[ATTACK_IDXS[0]] = adv_feats[i, 0]
        state_adv = state_adv.reshape(state_orig.shape)

        # state_orig = torch.tensor(state_orig, device=DEVICE).requires_grad_(True)
        # state_adv = torch.tensor(state_adv, device=DEVICE).requires_grad_(True)

        p_orig = np.argmax(model.predict(state_orig))
        p_adv = np.argmax(model.predict(state_adv))

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