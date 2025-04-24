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
#CSV_PATH      = "filtered_good_testing_data.csv"
CSV_PATH      = "test.csv"
SPEC_PATH     = "qoe_spec.json"
OUT_CSV       = "pgd_attacked_data.csv"


FEATURES = [                         # exactly five columns we attack
    "Last1_chunk_bitrate",
    "Last1_buffer_size",
    "Last1_downloadtime",
    "Last2_downloadtime",
    "Last3_downloadtime",
]

EPS   = 0.01
ALPHA = 0.001
STEPS = 40
TARGET_CLASS = 0                    # 300 kbps
brs = [300, 750, 1200, 1850, 2850, 4300]
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

    s[3, 2] = random.uniform(0.3,0.7)
    return s

def find_target_br(cur_br):
    """Find the target bitrate for the attack."""
    # Find the index of the target bitrate in the spec
    if cur_br[0] == brs[0]:
        return 0
    elif cur_br[0] == brs[1]:
        return 1-1
    elif cur_br[0] == brs[2]:
        return 2-1
    elif cur_br[0] == brs[3]:
        return 3-1
    elif cur_br[0] == brs[4]:
        return 4-1
    elif cur_br[0] == brs[5]:
        return 5-1
    
    # should not reach here
    return -1

# ============ 1. load CSV & spec ==================================

df   = pd.read_csv(CSV_PATH)
raw = json.load(open(SPEC_PATH))
specs = raw if isinstance(raw, list) else [raw]


low  = np.array([spec[f"{k}_l"] for k in FEATURES], dtype=np.float32)
high = np.array([spec[f"{k}_u"] for k in FEATURES], dtype=np.float32)

br_output_bound = np.array([spec["cur_br"]])

TARGET_CLASS = find_target_br(br_output_bound)
print(f"Target bitrate class : {TARGET_CLASS} ({brs[TARGET_CLASS]} kbps)")


states_np = np.stack([row_to_state(r) for _, r in df.iterrows()])  # (N,6,8)
orig_feats = df[FEATURES].values.astype(np.float32)                # (N,5)

N = states_np.shape[0]

# ============ 2. rebuild graph & PGD nodes ========================

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph(CKPT_PREFIX + ".meta")

    state_ph  = graph.get_tensor_by_name("actor/InputData/X:0")      # (None,6,8)
    logits_op = graph.get_tensor_by_name("actor/FullyConnected_4/MatMul:0")  # (None,6)

    # placeholders for the five mutable features
    x_ph      = tf.placeholder(tf.float32, shape=[None, 5], name="x_adv")

    # scatter x_ph back into a flat 48-vector ----------------------
    ATTACK_IDXS = [0, 8, 24, 25, 26]                       # flat indices
    idx_tf   = tf.constant([[0, i] for i in ATTACK_IDXS], dtype=tf.int64)

    state_flat = tf.reshape(state_ph, [-1, 48])            # (N,48)
    patched_flat = tf.tensor_scatter_nd_update(state_flat, idx_tf, x_ph[0])
    patched_state = tf.reshape(patched_flat, [-1, 6, 8])

    # forward pass on patched state
    logits_adv = logits_op                                   # reuse existing op
    # cross-entropy loss toward class-0
    target_onehot = tf.one_hot([TARGET_CLASS], 6)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_adv, labels=target_onehot)
    grad = tf.gradients(loss, x_ph)[0]                       # gradient w.r.t 5-features

# ============ 3. run PGD ==========================================

adv_feats = np.empty_like(orig_feats)

with tf.Session(graph=graph) as sess:
    saver.restore(sess, CKPT_PREFIX)


    for i in range(N):
        state0 = states_np[i:i+1]                 # keep batch dim =1
        x0     = orig_feats[i]                    # (5,)

        # skip if already class-0
        pred = sess.run(logits_op,
                        feed_dict={state_ph: state0})[0].argmax()
        if pred == TARGET_CLASS:
            adv_feats[i] = x0
            continue

        x_adv = x0.copy()

        orig_logits = sess.run(logits_op, {state_ph: state0})
        orig_probs  = sess.run(tf.nn.softmax(logits_op),
                           {state_ph: state0})[0]   # shape (6,)


        for _ in range(STEPS):
            # g = sess.run(grad, {x_ph: x_adv[None], state_ph: state0})[0]
            # x_adv -= ALPHA * np.sign(g)
            # x_adv = np.clip(x_adv, x0 - EPS, x0 + EPS)   # L∞ ball
            # x_adv = np.clip(x_adv, low, high)            # spec bounds
            # 1) run forward AND backward in one go
            grad_val, logits_val = sess.run(
                [grad, logits_op],
                feed_dict={state_ph: state0,      # full state
                        x_ph    : x_adv[None]} # (1,5) adv features
            )
            logits_val = logits_val[0]      # drop batch dim → shape (6,)
            grad_val   = grad_val[0, ROW_IDX, COL_IDX]  # shape (5,)

            # 2) Optional: inspect current prediction or loss
            pred_class = np.argmax(logits_val)
            curr_loss  = -logits_val[0,TARGET_CLASS]  # if you were using targeted CE
            print(f"Step {step:02d} | pred = {pred_class} | loss = {curr_loss:.4f}")

            # 3) PGD update + projections
            x_adv -= ALPHA * np.sign(grad_val)
            x_adv  = np.clip(x_adv, x0 - EPS, x0 + EPS)   # L∞
            x_adv  = np.clip(x_adv, low, high)            # spec bounds
            
        
        adv_feats[i] = x_adv

# ============ 4. write CSV ========================================

df.loc[:, FEATURES] = adv_feats
df.to_csv(OUT_CSV, index=False)
print(f"Adversarial CSV written →  {OUT_CSV}")


# ============ 5. quick verification =================================
print("\n▶  Verifying attack effectiveness …")

success  = 0          # adversarial predicts a *lower* class than original
unchanged = 0
higher   = 0
orig_preds = []
adv_preds  = []

with tf.Session(graph=graph) as sess:
    saver.restore(sess, CKPT_PREFIX)

    for i in range(N):
        state_orig = states_np[i:i+1]                    # (1,6,8)
        state_adv  = df.iloc[i]                          # row with adv feats
        # rebuild full 6×8 matrix with the five perturbed values
        state_adv = row_to_state(state_adv) [None]       # (1,6,8)

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
print("Original :", dict(sorted(C.Counter(orig_preds).items())))
print("Adversarial :", dict(sorted(C.Counter(adv_preds).items())))
