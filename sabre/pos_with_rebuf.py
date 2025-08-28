import os
import csv
from datetime import datetime
from collections import deque
from typing import Any, List

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from sabre import Abr, AbrInput  # Sabre classes

try:  # Attempt to import Sabre's get_buffer_level and manifest
    from sabre import get_buffer_level, manifest
except Exception:
    get_buffer_level = None  # type: ignore
    manifest = None  # type: ignore

# ------------------------------ Pensieve input parameters --------------------------------------
_S_INFO = 6            # state rows
_S_LEN = 8             # history window
_A_DIM = 6             # bitrate choices
_BUFFER_NORM = 10.0
_CHUNK_REMAIN_CAP = 48.0
_MODEL_CKPT = os.path.expanduser(
    "../pensieve_rl_model/pretrain_linear_reward.ckpt")

# tensor names in the Pensieve model
_TENSOR_IN  = "actor/InputData/X:0"
_TENSOR_OUT = "actor/FullyConnected_4/Softmax:0"

# Pensieve bitrate levels (kbps)
BRS = [300, 750, 1200, 1850, 2850, 4300]

# QoE formula constants
R_MIN = 216            # kbps for log normalization
REBUF_PENALTY = 4.3    # penalty per second of rebuffer
SMOOTH_PENALTY = 1.0   # penalty per bitrate switch

def g(R_kbps):
    """Log-normalized video quality function."""
    return np.log(R_kbps / R_MIN)

# --------------------------- Sabre ABR Implementation ----------------------------------
class pos(Abr):
    """Sabre wrapper for a pre‑trained Pensieve policy with QoE logging."""

    def __init__(self, config):
        super().__init__(config)

        # 1) Create session and load graph
        self._sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                                      inter_op_parallelism_threads=1))
        saver = tf.train.import_meta_graph(_MODEL_CKPT + ".meta")
        saver.restore(self._sess, _MODEL_CKPT)
        g_tf = tf.get_default_graph()
        self._s_in  = g_tf.get_tensor_by_name(_TENSOR_IN)
        self._q_out = g_tf.get_tensor_by_name(_TENSOR_OUT)

        # 2) History buffer
        self._last_q_hist = deque([0.0] * _S_LEN, maxlen=_S_LEN)
        self._buffer_hist = deque([0.0] * _S_LEN, maxlen=_S_LEN)
        self._thrpt_hist  = deque([1.0] * _S_LEN, maxlen=_S_LEN)
        self._delay_hist  = deque([0.4] * _S_LEN, maxlen=_S_LEN)
        self._next_sizes  = np.zeros(_A_DIM, dtype=np.float32)
        self._remain_hist = deque([1.0] * _S_LEN, maxlen=_S_LEN)

        self._last_quality = 0  # Last decision
        self._starting_bitrate = int(config.get("starting_bitrate", 0))

        # 🔥 CHANGED: QoE tracking
        self._cumulative_qoe = 0.0
        self._last_gR = None  # For smoothness penalty

        # 🔥 CHANGED: Setup CSV logging with avg QoE
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, f"pensieve_log_{timestamp}.csv")
        self._log_writer = csv.writer(open(self._log_file, "w", newline=""))
        self._log_writer.writerow(["segment_index", "bitrate_kbps", "download_time_ms", "rebuffer_s", "cumulative_qoe", "average_qoe"])

        print(f"📄 Logging to: {self._log_file}")

    def get_first_quality(self):
        return self._starting_bitrate  # index 0~5

    def get_quality_delay(self, segment_index: int):
        """Return (quality_index, extra_delay_ms)."""
        state_mat = np.vstack([
            np.array(self._last_q_hist),
            np.array(self._buffer_hist),
            np.array(self._thrpt_hist),
            np.array(self._delay_hist),
            np.pad(self._next_sizes, (0, _S_LEN - _A_DIM)),
            np.array(self._remain_hist),
        ], dtype=np.float32)

        probs = self._sess.run(self._q_out, {self._s_in: state_mat[np.newaxis, ...]})[0]
        quality = int(np.argmax(probs))
        self._last_quality = quality
        return quality, 0.0

    def report_download(self, metrics: Any, is_replacement: bool):  # type: ignore
        """metrics ≡ sabre.DownloadProgress namedtuple."""
        delay_ms   = max(metrics.time, 1e-3)
        bytes_dl   = metrics.downloaded
        seg_idx    = metrics.index

        kb_per_ms = bytes_dl / 1000.0 / delay_ms

        if get_buffer_level is not None:
            buffer_ms = get_buffer_level()
            buffer_sec = buffer_ms / 1000.0
        else:
            buffer_sec = 0.0

        if manifest is not None and hasattr(manifest, 'segments'):
            remain = max(len(manifest.segments) - seg_idx - 1, 0)
        else:
            remain = 0

        if manifest is not None and seg_idx + 1 < len(manifest.segments):
            next_sizes_bits = manifest.segments[seg_idx + 1]
            next_sizes_mb = np.array(next_sizes_bits[:_A_DIM], dtype=np.float32) / 8.0 / 1e6
        else:
            next_sizes_mb = np.zeros(_A_DIM, dtype=np.float32)

        # Update history
        self._last_q_hist.append(float(self._last_quality) / (_A_DIM - 1))
        self._buffer_hist.append(buffer_sec / _BUFFER_NORM)
        self._thrpt_hist.append(kb_per_ms)
        self._delay_hist.append(delay_ms / 10000.0)
        self._remain_hist.append(min(remain, _CHUNK_REMAIN_CAP) / _CHUNK_REMAIN_CAP)
        self._next_sizes = next_sizes_mb

        # 🔥 CHANGED: QoE calculation with rebuffer penalty
        bitrate_kbps = BRS[self._last_quality]
        gR = g(bitrate_kbps)

        # Rebuffer time (seconds)
        rebuffer_s = max(0.0, (delay_ms / 1000.0) - buffer_sec)

        reward = gR
        reward -= REBUF_PENALTY * rebuffer_s
        if self._last_gR is not None:
            smoothness_penalty = abs(gR - self._last_gR)
            reward -= SMOOTH_PENALTY * smoothness_penalty

        self._cumulative_qoe += reward
        self._last_gR = gR

        # 🔥 CHANGED: Average QoE
        average_qoe = self._cumulative_qoe / (seg_idx + 1)

        # 🔥 CHANGED: Log this chunk with avg QoE
        self._log_writer.writerow([seg_idx, bitrate_kbps, delay_ms, rebuffer_s, self._cumulative_qoe, average_qoe])
        print(f"[Chunk {seg_idx}] Bitrate={bitrate_kbps} kbps, DL={delay_ms:.1f} ms, Rebuffer={rebuffer_s:.2f} s, QoE={self._cumulative_qoe:.3f}, AvgQoE={average_qoe:.3f}")

    def report_delay(self, delay):
        pass

    def report_seek(self, where):
        pass
