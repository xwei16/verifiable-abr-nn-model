#!/usr/bin/env python3
# pgd_tf_random_5feat.py  ------------------------------------------
"""
PGD attack on five Pensieve features (TF-1.x):

• Synthetic batch of M states ∈ [0,1]
• Last1_chunk_bitrate fixed at its upper bound
• Mutable features: indices [0,8,24,25,26] in the 6×8 state
• Loss: expected bitrate under the model’s softmax
• Projection: L∞≤ε plus spec bounds
• Output: pgd_attacked_data.csv + success stats
"""

import os, json, random, collections as C
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────
CKPT_PREFIX   = "/home/xwei16/abr-verification/pensieve_rl_model/pretrain_linear_reward.ckpt"
SPEC_PATH     = "qoe_spec.json"
OUT_CSV       = "pgd_attacked_data.csv"

M             = 200             # synthetic batch size
EPS           = 0.01
ALPHA         = 0.001
STEPS         = 40
# flat indices in the 6×8 state for the five features:
ATTACK_IDXS   = [0, 8, 24, 25, 26]
ROW_IDX, COL_IDX = zip(*[divmod(i, 8) for i in ATTACK_IDXS])
# real bitrate ladder (kbps)
BRATES        = np.array([300., 750., 1200., 1850., 2850., 4300.], np.float32)
# ────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════
# 0. load spec and define helpers
# ═══════════════════════════════════════════════════════════════════
spec = json.load(open(SPEC_PATH))
if isinstance(spec, list):
    spec = spec[0]

low  = np.array([spec["Last1_chunk_bitrate_l"],
                 spec["Last1_buffer_size_l"],
                 spec["Last1_downloadtime_l"],
                 spec["Last2_downloadtime_l"],
                 spec["Last3_downloadtime_l"]], np.float32)
high = np.array([spec["Last1_chunk_bitrate_u"],
                 spec["Last1_buffer_size_u"],
                 spec["Last1_downloadtime_u"],
                 spec["Last2_downloadtime_u"],
                 spec["Last3_downloadtime_u"]], np.float32)

def build_state():
    """Return one (6,8) state with all entries ∈ [0,1], but
       Last1_chunk_bitrate fixed at its upper-bound."""
    s = np.random.uniform(0.0, 1.0, (6, 8)).astype(np.float32)
    s[0, 0] = spec["Last1_chunk_bitrate_u"]        # highest bitrate
    # buffer in its own bounds
    s[1, 0] = random.uniform(spec["Last1_buffer_size_l"],
                              spec["Last1_buffer_size_u"])
    # download-time history: three bounds, five free
    s[3, 0] = random.uniform(spec["Last1_downloadtime_l"],
                              spec["Last1_downloadtime_u"])
    s[3, 1] = random.uniform(spec["Last2_downloadtime_l"],
                              spec["Last2_downloadtime_u"])
    s[3, 2] = random.uniform(spec["Last3_downloadtime_l"],
                              spec["Last3_downloadtime_u"])
    s[3, 3:] = np.random.uniform(0.0, 1.0, size=5)
    return s

def state_to_row(s):
    """Flatten a (6,8) state into the five attacked columns for CSV."""
    return {
        "Last1_chunk_bitrate": s[0, 0],
        "Last1_buffer_size"  : s[1, 0],
        "Last1_downloadtime" : s[3, 0],
        "Last2_downloadtime" : s[3, 1],
        "Last3_downloadtime" : s[3, 2],
    }

# ═══════════════════════════════════════════════════════════════════
# 1. generate synthetic batch
# ═══════════════════════════════════════════════════════════════════
states_np = np.stack([build_state() for _ in range(M)])   # (M,6,8)
adv_states = states_np.copy()
N = M
print(f"▶ Generated {N} synthetic states; running PGD…")

# ═══════════════════════════════════════════════════════════════════
# 2. rebuild graph & define loss/gradient
# ═══════════════════════════════════════════════════════════════════
graph = tf.Graph()
with graph.as_default():
    saver     = tf.train.import_meta_graph(CKPT_PREFIX + ".meta")
    state_ph  = graph.get_tensor_by_name("actor/InputData/X:0")           # (None, 6, 8)
    logits_op = graph.get_tensor_by_name("actor/FullyConnected_4/MatMul:0")  # (None, 6)

    # turn logits → probabilities
    probs     = tf.nn.softmax(logits_op)                                  # (None,6)
    # expected bitrate under the model
    expected_br = tf.reduce_sum(probs * BRATES, axis=1)                  # (None,)

    # our loss: make expected_br as *small* as possible
    loss      = expected_br
    # gradient of that scalar loss w.r.t. the full input state
    grad_state = tf.gradients(loss, state_ph)[0]                         # (None,6,8)

# ═══════════════════════════════════════════════════════════════════
# 3. run PGD over the five coordinates only
# ═══════════════════════════════════════════════════════════════════
with tf.Session(graph=graph) as sess:
    saver.restore(sess, CKPT_PREFIX)

    for i in range(N):
        s_adv = adv_states[i : i + 1]          # shape (1,6,8)

        # skip if already outputting the lowest bitrate (index 0)
        orig_pred = sess.run(logits_op, {state_ph: s_adv}).argmax()
        if orig_pred == 0:
            continue

        # record the *original* five features
        x0 = s_adv[0, ROW_IDX, COL_IDX].copy()   # shape (5,)

        # PGD iterations
        for step in range(STEPS):
            g_state = sess.run(grad_state, {state_ph: s_adv})  # (1,6,8)
            g5      = g_state[0, ROW_IDX, COL_IDX]            # (5,)

            # 1) step in sign(gradient)
            x5 = s_adv[0, ROW_IDX, COL_IDX] - ALPHA * np.sign(g5)

            # 2) project into L∞ ball around x0
            x5 = np.clip(x5, x0 - EPS, x0 + EPS)

            # 3) enforce physical specs
            x5 = np.clip(x5, low, high)

            # write back just those five coords
            s_adv[0, ROW_IDX, COL_IDX] = x5

        # store the final adversarial state
        adv_states[i] = s_adv

# ═══════════════════════════════════════════════════════════════════
# 4. write CSV of attacked columns
# ═══════════════════════════════════════════════════════════════════
df_out = pd.DataFrame([state_to_row(adv_states[i]) for i in range(N)])
df_out.to_csv(OUT_CSV, index=False)
print(f"▶ Adversarial CSV written → {OUT_CSV}")

# ═══════════════════════════════════════════════════════════════════
# 5. quick verification
# ═══════════════════════════════════════════════════════════════════
succ = same = up = 0
orig_p = []
adv_p  = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, CKPT_PREFIX)
    for i in range(N):
        p_o = sess.run(logits_op, {state_ph: states_np[i:i+1]}).argmax()
        p_a = sess.run(logits_op, {state_ph: adv_states[i:i+1]}).argmax()

        orig_p.append(int(p_o)); adv_p.append(int(p_a))
        if p_a < p_o:  succ += 1
        elif p_a == p_o: same += 1
        else:           up   += 1

print("\n▶ Verification")
print(f" Total samples           : {N}")
print(f" Successful downgrades   : {succ} ({succ/N*100:.1f}%)")
print(f" Unchanged predictions   : {same}")
print(f" Accidentally higher br  : {up}")
print("\nBit-rate histograms (index 0=300…5=4300)")
print(" Original:", dict(sorted(C.Counter(orig_p).items())))
print(" Adversarial:", dict(sorted(C.Counter(adv_p).items())))
