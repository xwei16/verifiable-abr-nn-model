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
import time

from tqdm import tqdm

import ppo2 as network

from utils import load_ppo2_model, row_to_state, row_to_state_for_rand


MODEL_PATH   = "pensieve_rl_model/nn_model_ep_155400.pth"
#CSV_PATH     = "data/pensieve_big_testing_data.csv"
# CSV_PATH     = "../../src/filtered_good_testing_data.csv"
SPEC_PATH    = "data/full_spec_no_300.json"
OUT_CSV      = "results/rand_pgd_attacked_torch_data.csv"
OUT_SUCCESSFUL_CSV = "results/rand_pgd_successful_attack.csv"

FEATURES = [                         # feature(s) to attack
    "Last1_downloadtime"
]
ATTACK_IDXS = [31]

EPS   = 0.01                        # epsilon for L∞ bound
ALPHA = 0.001                       # step size
STEPS = 40                          # number of PGD steps
TARGET_CLASS = 0                    # target class [0: 300 kbps, 1: 1000 kbps]
M = 0                               # number of spec
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# PPO2 parameters
S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4

NUM_CONT_CHUNKS_TO_LOG = 1


ALL_FEATURES = [
    'Last1_chunk_bitrate', 'Last1_buffer_size', 'Last8_throughput', 'Last7_throughput',
    'Last6_throughput', 'Last5_throughput', 'Last4_throughput', 'Last3_throughput',
    'Last2_throughput', 'Last1_throughput', 'Last8_downloadtime', 'Last7_downloadtime',
    'Last6_downloadtime', 'Last5_downloadtime', 'Last4_downloadtime', 'Last3_downloadtime',
    'Last2_downloadtime', 'Last1_downloadtime', 'chunksize1', 'chunksize2', 'chunksize3',
    'chunksize4', 'chunksize5', 'chunksize6', 'Chunks_left', 
    "br","qoe_2"
]

NUM_TOTAL_FEATURES = len(ALL_FEATURES)


df = pd.DataFrame(columns=ALL_FEATURES)

raw  = json.load(open(SPEC_PATH))
spec  = raw if isinstance(raw, list) else [raw]

M    = len(spec)
N = 30000

model = load_ppo2_model(MODEL_PATH, S_DIM, A_DIM, ACTOR_LR_RATE)

## PGD attack

adv_feats = np.zeros((N, 1))

def generate_random_data_row(spec):
    """Generate a random data row from the spec."""
    random.seed(int(time.time()))
    chosen = spec[random.randint(0, len(spec) - 1)]

    row = []
    for i in range(NUM_TOTAL_FEATURES-2):
        row.append(random.uniform(
        chosen[f"{ALL_FEATURES[i]}_l"],
        chosen[f"{ALL_FEATURES[i]}_u"]
    ))
    #br
    row.append(random.choice(chosen["cur_br"]))
    #qoe_2
    row.append(random.choice(chosen["qoe_2"]))
        
    df.loc[len(df)] = row

    return chosen["Last1_downloadtime_l"], chosen["Last1_downloadtime_u"]


for i in tqdm(range(N), desc="PGD attack"):
    # Generate random data row
    low_bound, high_bound = generate_random_data_row(spec)

    state0 = row_to_state_for_rand(df.iloc[i])
    x0 = state0[3, 7].astype(np.float32)

    state0 = np.expand_dims(state0, axis=0)
    pred = model.predict(state0)
    pred = np.argmax(pred)

    
    if pred == TARGET_CLASS:
        adv_feats[i] = x0
        continue
    
    x_adv = x0
    x_adv_prev = x0
    state_cur = None
    thrpt_times = None
    
    for _ in range(STEPS):
        state_cur = state0.flatten()
        state_cur[ATTACK_IDXS[0]] = float(x_adv)
        thrpt_times = x_adv_prev / x_adv
        state_cur[ATTACK_IDXS[0] - 8] *= thrpt_times
        state_cur = state_cur.reshape(state0.shape)

        state_cur = torch.from_numpy(state_cur).to(torch.float32).requires_grad_(True)

        output = model.forward(state_cur)
        # Add batch dimension
        if output.dim() == 1:
            output = output.unsqueeze(0)    # Now shape: (1, 6)
        target = torch.tensor([TARGET_CLASS], device=DEVICE)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        grad = state_cur.grad.view(-1)[ATTACK_IDXS[0]]

        x_perturbation = ALPHA * torch.sign(grad)
        x_adv_prev = x_adv
        x_adv = x_perturbation
        x_adv = torch.clamp(x_adv, x0 - EPS, x0 + EPS)
        x_adv = torch.clamp(x_adv, low_bound, high_bound)
    
    adv_feats[i] = x_adv


## Verify attack effectiveness

def write_success_row_to_csv(df, i):
    if i == 0:
        #write header
        df.iloc[i].to_csv(OUT_SUCCESSFUL_CSV, mode='w', header=True, index=False)
    else:
        row = df.iloc[i].copy()
        # write original row to CSV
        row.to_csv(OUT_SUCCESSFUL_CSV, mode='a', header=False, index=False)
        # write row to CSV
        row[FEATURES] = adv_feats[i]
        row.to_csv(OUT_SUCCESSFUL_CSV, mode='a', header=False, index=False)
        #write an empty row
        dash_line = ['---'] * NUM_TOTAL_FEATURES
        pd.DataFrame([dash_line]).to_csv(OUT_SUCCESSFUL_CSV, mode='a', header=False, index=False)

    pass

success = unchanged = higher = 0
orig_preds, adv_preds = [], []

for i in tqdm(range(N), desc="Verifying attack effectiveness"):
    state_orig = row_to_state_for_rand(df.iloc[i])
    
    x_adv_prev = state_orig[3, 7].astype(np.float32)

    state_orig_flat = state_orig.flatten()
    state_adv = state_orig_flat.copy()
    thrpt_times = x_adv_prev / adv_feats[i]
    state_adv[ATTACK_IDXS[0]] = adv_feats[i]
    state_adv[ATTACK_IDXS[0] - 8] *= thrpt_times
    state_adv = state_adv.reshape(state_orig.shape)

    state_orig = np.expand_dims(state_orig, axis=0)
    state_adv = np.expand_dims(state_adv, axis=0)

    p_orig = np.argmax(model.predict(state_orig))
    p_adv = np.argmax(model.predict(state_adv))

    orig_preds.append(p_orig)
    adv_preds.append(p_adv)

    if p_adv < p_orig:
        success += 1
        write_success_row_to_csv(df, i)
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


## Write to CSV

adv_feats = adv_feats.reshape(-1, 1)  # shape (N, 1)
df.loc[:, FEATURES] = adv_feats
df.to_csv(OUT_CSV, index=False)
print(f"Adversarial CSV written →  {OUT_CSV}")
