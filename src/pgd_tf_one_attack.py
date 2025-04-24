#!/usr/bin/env python3
# pgd_pensieve_tf.py  ----------------------------------------------
"""
Targeted PGD against Pensieve’s actor network
  • pushes every state toward class-0 (300 kbps)
  • perturbs only five features and respects both an L∞ budget (eps)
    and the physical bounds in qoe_spec.json
Outputs a new CSV "pgd_attacked_data.csv" whose five columns have been
adversarially modified; all other columns stay identical.
"""

import json
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random

# -------------- CONFIG --------------------------------------------
CKPT_PREFIX   = "/home/xwei16/abr-verification/pensieve_rl_model/pretrain_linear_reward.ckpt"
CSV_PATH      = "filtered_good_testing_data.csv"#"pensieve_big_testing_data.csv"
SPEC_PATH     = "qoe_spec.json"
OUT_CSV       = "pgd_attacked_data.csv"

FEATURES = [                         # exactly five columns we attack
    "Last1_downloadtime"
]

EPS   = 0.01
ALPHA = 0.001
STEPS = 40
TARGET_CLASS = 0                    # 300 kbps
M = 0                                # will be initialized later
# ------------------------------------------------------------------

# ============ 0. helpers ==========================================

def row_to_state(r):
    """Convert one pandas row to Pensieve state (6×8)."""
    s = np.zeros((6, 8), dtype=np.float32)
    s[0, 0] = r["Last1_chunk_bitrate"]
    s[1, 0] = r["Last1_buffer_size"]
    s[2, :] = r[[f"Last{i}_throughput"   for i in range(8, 0, -1)]].values
    s[3, :] = r[[f"Last{i}_downloadtime" for i in range(8, 0, -1)]].values
    s[4, :6] = r[[f"chunksize{i}"        for i in range(1, 7)]].values
    s[5, 0] = r["Chunks_left"]
    chosen = random.choice(spec)
    s[3, 7] = random.uniform(
        chosen["Last1_downloadtime_l"],
        chosen["Last1_downloadtime_u"]
    )
    return s


# ============ 1. load CSV & spec ==================================

df   = pd.read_csv(CSV_PATH)
raw  = json.load(open(SPEC_PATH))
spec = raw if isinstance(raw, list) else [raw]
M    = len(spec)

low  = np.array([spec[i][f"{k}_l"] for k in FEATURES for i in range(M)],
                dtype=np.float32)
high = np.array([spec[i][f"{k}_u"] for k in FEATURES for i in range(M)],
                dtype=np.float32)

states_np  = np.stack([row_to_state(r) for _, r in df.iterrows()])  # (N,6,8)
# need to be changed later!!!
orig_feats = np.array([states_np[i][3,7].astype(np.float32) for i in range(len(states_np))])               # (N,1)
N          = states_np.shape[0]

# ============ 2. rebuild graph & PGD nodes ========================

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph(CKPT_PREFIX + ".meta")

    state_ph  = graph.get_tensor_by_name("actor/InputData/X:0")      # (None,6,8)
    logits_op = graph.get_tensor_by_name("actor/FullyConnected_4/MatMul:0")  # (None,6)

    ATTACK_IDXS = [31]                       # flat indices (row-3, col-0) for Last1_downloadtime

    # cross-entropy loss toward class-0
    target_onehot = tf.one_hot([TARGET_CLASS], 6)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_op, labels=target_onehot)

    # gradient of loss w.r.t. the *whole* state tensor  ###############
    grad_state = tf.gradients(loss, state_ph)[0]            ### <<< changed


# ============ 3. run PGD ==========================================

adv_feats = np.empty_like(orig_feats)

with tf.Session(graph=graph) as sess:
    saver.restore(sess, CKPT_PREFIX)

    for i in range(N):
        state0 = states_np[i:i+1]                 # keep batch dim = 1
        x0     = orig_feats[i]                 # scalar (one feature)

        # skip if already class-0
        pred = sess.run(logits_op,
                        feed_dict={state_ph: state0})[0].argmax()
        if pred == TARGET_CLASS:
            adv_feats[i] = x0
            continue

        x_adv = x0
        for _ in range(STEPS):
            # inject current adversarial value into a copy of the state
            state_cur = state0.copy()                                      ### <<< added
            state_cur.flat[ATTACK_IDXS[0]] = float(x_adv)   # changed                     ### <<< added

            # gradient for that element
            g_state = sess.run(grad_state, {state_ph: state_cur})[0]       ### <<< changed
            g       = g_state.flat[ATTACK_IDXS[0]]                         ### <<< added

            x_adv -= ALPHA * np.sign(g)
            x_adv  = np.clip(x_adv, x0 - EPS, x0 + EPS)   # L∞ ball
            x_adv  = np.clip(x_adv, low, high)            # spec bounds

        adv_feats[i] = x_adv

# ============ 4. write CSV ========================================

adv_feats = adv_feats.reshape(-1, 1)  # shape (43, 1)   - changed
df.loc[:, FEATURES] = adv_feats
df.to_csv(OUT_CSV, index=False)
print(f"Adversarial CSV written →  {OUT_CSV}")

# ============ 5. quick verification ===============================

print("\n▶  Verifying attack effectiveness …")

success = unchanged = higher = 0
orig_preds, adv_preds = [], []

with tf.Session(graph=graph) as sess:
    saver.restore(sess, CKPT_PREFIX)

    for i in range(N):
        state_orig = states_np[i:i+1]                    # (1,6,8)
        state_adv  = state_orig.copy()
        state_adv.flat[ATTACK_IDXS[0]] = adv_feats[i, 0] ### <<< changed

        p_orig = sess.run(logits_op, {state_ph: state_orig}).argmax()
        p_adv  = sess.run(logits_op, {state_ph: state_adv }).argmax()

        orig_preds.append(p_orig)
        adv_preds.append(p_adv)

        if p_adv < p_orig:
            success += 1
        elif p_adv == p_orig:
            unchanged += 1
        else:
            higher += 1

print(f"Total samples           : {N}")
print(f"Successful downgrades   : {success}  "
      f"({success / N * 100:.1f}%)")
print(f"Unchanged predictions   : {unchanged}")
print(f"Accidentally higher br  : {higher}")

# Optional: distribution before/after
import collections as C
print("\nBit-rate histogram (index 0-5)")
print("Original    :", dict(sorted(C.Counter(orig_preds).items())))
print("Adversarial :", dict(sorted(C.Counter(adv_preds).items())))
