#!/usr/bin/env python3
# pgd_tf_random.py  -------------------------------------------------
"""
Targeted PGD against Pensieve (TF-1.x) – synthetic data variant
▹ one mutable feature  : Last1_downloadtime  (flat idx 24)
▹ bitrate fixed high   : Last1_chunk_bitrate = upper bound
▹ ε-ball (L∞)          : 0.01  •  40 steps
▹ bounds respected     : per-feature limits from qoe_spec.json
Outputs: pgd_attacked_data.csv  and success statistics
"""

import json, random, collections as C, os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────
CKPT_PREFIX = "/home/xwei16/abr-verification/pensieve_rl_model/pretrain_linear_reward.ckpt"
SPEC_PATH   = "qoe_spec.json"
OUT_CSV     = "pgd_attacked_data.csv"

M                = 200          # synthetic batch size
ATTACK_IDX_FLAT  = 24           # (row 3, col 0) in 6×8 matrix
EPS     = 0.01
ALPHA   = 0.001
STEPS   = 40
TARGET_CLASS = 0                # 300 kbps
# ────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════
# 0. load bounds & helpers
# ═══════════════════════════════════════════════════════════════════
spec = json.load(open(SPEC_PATH))
if isinstance(spec, list):
    spec = spec[0]

low  = spec["Last1_downloadtime_l"]
high = spec["Last1_downloadtime_u"]

def rand_between(key):
    return random.uniform(spec[f"{key}_l"], spec[f"{key}_u"])

def build_random_state():
    """Return one (6,8) normalised state inside bounds."""
    s = np.zeros((6, 8), np.float32)
    s[0, 0] = spec["Last1_chunk_bitrate_u"]      # highest quality
    s[1, 0] = rand_between("Last1_buffer_size")  # buffer
    s[2, :] = np.random.uniform(0.0, 1.0, 8)     # throughput hist
    s[3, 0] = rand_between("Last1_downloadtime") # the attacked cell
    s[3, 1:] = np.clip(np.random.normal(s[3, 0], 0.02, 7), 0.0, 1.0)
    s[4, :6] = np.random.uniform(0.0, 1.0, 6)    # chunk sizes
    s[5, 0]  = np.random.uniform(0.0, 1.0)       # chunks left
    return s

def state_to_series(s):
    """Flatten one (6,8) state into a dict compatible with CSV fields."""
    r = {"Last1_chunk_bitrate": s[0,0],
         "Last1_buffer_size"  : s[1,0],
         "Chunks_left"        : s[5,0]}
    r.update({f"Last{i}_throughput"   : s[2, 8-i] for i in range(8,0,-1)})
    r.update({f"Last{i}_downloadtime" : s[3, 8-i] for i in range(8,0,-1)})
    r.update({f"chunksize{i}"         : s[4, i-1] for i in range(1,7)})
    return r

# ═══════════════════════════════════════════════════════════════════
# 1. synthetic dataset
# ═══════════════════════════════════════════════════════════════════
states_np = np.stack([build_random_state() for _ in range(M)])       # (M,6,8)
N = states_np.shape[0]
print(f"Generated {N} synthetic states – starting PGD …")

# ═══════════════════════════════════════════════════════════════════
# 2. restore graph – take gradient w.r.t. *state_ph*
# ═══════════════════════════════════════════════════════════════════
graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph(CKPT_PREFIX + ".meta")

    state_ph  = graph.get_tensor_by_name("actor/InputData/X:0")          # (None,6,8)
    logits_op = graph.get_tensor_by_name("actor/FullyConnected_4/MatMul:0")

    target_onehot = tf.one_hot([TARGET_CLASS], 6)
    loss  = tf.nn.softmax_cross_entropy_with_logits(logits=logits_op,
                                                    labels=target_onehot)
    grad_state = tf.gradients(loss, state_ph)[0]     # (None,6,8)

# ═══════════════════════════════════════════════════════════════════
# 3. run PGD
# ═══════════════════════════════════════════════════════════════════
adv_states = states_np.copy()                         # will hold perturbed copies
idx_r, idx_c = divmod(ATTACK_IDX_FLAT, 8)             # (3,0)

with tf.Session(graph=graph) as sess:
    saver.restore(sess, CKPT_PREFIX)

    for i in range(N):
        s_adv = adv_states[i:i+1]                     # view (1,6,8)
        x0    = float(s_adv[0, idx_r, idx_c])         # scalar
        if sess.run(logits_op, {state_ph: s_adv}).argmax() == TARGET_CLASS:
            continue                                  # already class-0

        x_adv = x0
        for _ in range(STEPS):
            g = sess.run(grad_state, {state_ph: s_adv})[0, idx_r, idx_c]
            x_adv -= ALPHA * np.sign(g)
            x_adv  = np.clip(x_adv, x0 - EPS, x0 + EPS)   # ε-ball
            x_adv  = np.clip(x_adv, low, high)            # bounds
            s_adv[0, idx_r, idx_c] = x_adv               # in-place update

# ═══════════════════════════════════════════════════════════════════
# 4. write CSV
# ═══════════════════════════════════════════════════════════════════
rows = [state_to_series(adv_states[i]) for i in range(N)]
df   = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Adversarial CSV written →  {OUT_CSV}")

# ═══════════════════════════════════════════════════════════════════
# 5. verification
# ═══════════════════════════════════════════════════════════════════
success = unchanged = higher = 0
orig_preds = []
adv_preds  = []

with tf.Session(graph=graph) as sess:
    saver.restore(sess, CKPT_PREFIX)

    for i in range(N):
        s_orig = states_np[i:i+1]
        s_adv  = adv_states[i:i+1]

        p_orig = sess.run(logits_op, {state_ph: s_orig}).argmax()
        p_adv  = sess.run(logits_op, {state_ph: s_adv }).argmax()

        orig_preds.append(p_orig)
        adv_preds.append(p_adv)

        if p_adv < p_orig:
            success += 1
        elif p_adv == p_orig:
            unchanged += 1
        else:
            higher += 1

print("\n▶  Verification")
print(f"Total samples           : {N}")
print(f"Successful downgrades   : {success}  ({success/N*100:.1f}%)")
print(f"Unchanged predictions   : {unchanged}")
print(f"Accidentally higher br  : {higher}")

print("\nBit-rate histogram (0 = 300 kbps … 5 = 4300 kbps)")
print("Original :", dict(sorted(C.Counter(orig_preds).items())))
print("Adv      :", dict(sorted(C.Counter(adv_preds).items())))
