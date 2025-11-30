# python3 ../sabre/sabre_new.py -m ../sabre/traces/abr_test/pensieve_big.json -n ../sabre/traces/abr_test/network.json -a pensieve_pgd_pos.py -p starting_bitrate=3
# generated output are in logs
import os
import csv
from datetime import datetime
from collections import deque
from typing import Any, List

import numpy as np
import tensorflow.compat.v1 as tf

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

tf.disable_eager_execution()

from pgd_in_sabre import PGD # PGD attack
from sabre_new import Abr, AbrInput # Sabre classes

try:  # Attempt to import Sabre's get_buffer_level and manifest
    from sabre.sabre_new import get_buffer_level, manifest
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
    "../../model/abr-model/pensieve_rl_model/pretrain_linear_reward.ckpt")

# tensor names in the Pensieve model
_TENSOR_IN  = "actor/InputData/X:0"
_TENSOR_OUT = "actor/FullyConnected_4/Softmax:0"

# Pensieve bitrate levels (kbps)
BRS = [300, 750, 1200, 1850, 2850, 4300]

# QoE formula constants
R_MIN = 260  # kbps for log normalization

def g(R_kbps):
    """Log-normalized video quality function."""
    return np.log(R_kbps / R_MIN)

# --------------------------- Sabre ABR Implementation ----------------------------------
class pensieve_pgd_pos(Abr):
    """Sabre wrapper for a pre-trained Pensieve policy with QoE logging."""

    def __init__(self, config):
        super().__init__(config)

        # Create session and load graph
        self._sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                                      inter_op_parallelism_threads=1))
        saver = tf.train.import_meta_graph(_MODEL_CKPT + ".meta")
        saver.restore(self._sess, _MODEL_CKPT)
        g_tf = tf.get_default_graph()
        self._s_in  = g_tf.get_tensor_by_name(_TENSOR_IN)
        self._q_out = g_tf.get_tensor_by_name(_TENSOR_OUT)

        # History buffer
        self._last_q_hist = deque([0.0] * _S_LEN, maxlen=_S_LEN)
        self._buffer_hist = deque([0.0] * _S_LEN, maxlen=_S_LEN)
        self._thrpt_hist  = deque([1.0] * _S_LEN, maxlen=_S_LEN)
        self._delay_hist  = deque([0.4] * _S_LEN, maxlen=_S_LEN)
        self._next_sizes  = np.zeros(_A_DIM, dtype=np.float32)
        self._remain_hist = deque([1.0] * _S_LEN, maxlen=_S_LEN)

        self._last_quality = 0  # Last decision
        self._starting_bitrate = int(config.get("starting_bitrate", 3))

        # QoE tracking
        self._cumulative_qoe = 0.0
        # QoE calculation
        bitrate_kbps = BRS[self._last_quality]
        gR = g(bitrate_kbps)
        # Add quality term
        self._cumulative_qoe += gR
        self._last_gR = gR  # For smoothness penalty

        # Setup CSV logging with avg QoE
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, f"pensieve_log_{timestamp}.csv")
        self._log_writer = csv.writer(open(self._log_file, "w", newline=""))
        self._log_writer.writerow(["segment_index", "bitrate_kbps", "download_time_ms", "cumulative_qoe", "average_qoe"])

        print(f"📄 Logging to: {self._log_file}")

    def get_first_quality(self):
        return self._starting_bitrate  # index 0~5

    def get_quality_delay(self, segment_index: int):
        """Return (quality_index, extra_delay_ms)."""
        # load history state
        state_mat = np.vstack([
            np.array(self._last_q_hist),
            np.array(self._buffer_hist),
            np.array(self._thrpt_hist),
            np.array(self._delay_hist),
            np.pad(self._next_sizes, (0, _S_LEN - _A_DIM)),
            np.array(self._remain_hist),
        ], dtype=np.float32)

        # run Pensieve, probs store output - probs for each br
        probs = self._sess.run(self._q_out, {self._s_in: state_mat[np.newaxis, ...]})[0]
        
        # find br with max probability
        quality = int(np.argmax(probs))

        # update history
        self._last_quality = quality
        return quality, 0.0

    def report_download(self, metrics: Any, is_replacement: bool):  # type: ignore
        '''metrics ≡ sabre.DownloadProgress namedtuple.'''
        delay_ms   = max(metrics.time, 1e-3) # actual wall-clock time (in milliseconds) it took to download the last chunk.
        bytes_dl   = metrics.downloaded
        seg_idx    = metrics.index

        kb_per_ms = bytes_dl / 1000.0 / delay_ms # throughput based on downloading time

        if get_buffer_level is not None:
            buffer_ms = get_buffer_level() # returns the current player buffer size (in milliseconds).
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


        '''simulate adversarial attack on downloading time => change delay_ms'''
        attack = PGD()
        # load history state
        state = np.vstack([
            np.array(self._last_q_hist),
            np.array(self._buffer_hist),
            np.array(self._thrpt_hist),
            np.array(self._delay_hist),
            np.pad(self._next_sizes, (0, _S_LEN - _A_DIM)),
            np.array(self._remain_hist),
        ], dtype=np.float32)

        attacked_dt = attack.attack_download(state) * 10000.0
        attacked_thrpt = bytes_dl / 1000.0 / attacked_dt # throughput based on downloading time
        print(f"Original DT: {delay_ms:.1f} ms, Attacked DT: {attacked_dt:.1f} ms")
        self._thrpt_hist[-1] = attacked_thrpt
        self._delay_hist[-1] = (attacked_dt / 10000.0)

        # attack.attack_evaluation(state, attacked_dt, delay_ms)


        # QoE calculation
        bitrate_kbps = BRS[self._last_quality]
        gR = g(bitrate_kbps)

        # Add quality term
        self._cumulative_qoe += gR

        # Add smoothness penalty
        smoothness_penalty = 0.0
        if self._last_gR is not None:
            smoothness_penalty = abs(gR - self._last_gR)
            self._cumulative_qoe -= smoothness_penalty

        self._last_gR = gR

        # Calculate average QoE
        avg_qoe = self._cumulative_qoe / (seg_idx + 1)

        # Log this chunk with avg QoE
        self._log_writer.writerow([seg_idx, bitrate_kbps, delay_ms, self._cumulative_qoe, avg_qoe])
        print(f"[Chunk {seg_idx}] Bitrate={bitrate_kbps} kbps, Download Time={delay_ms:.1f} ms, QoE={self._cumulative_qoe:.3f}, AvgQoE={avg_qoe:.3f}")

    def report_delay(self, delay):
        pass

    def report_seek(self, where):
        pass
