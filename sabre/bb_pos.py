# bb_pos.py — Buffer-Based ABR for SABRE
# Run example (inside sabre directory):
#   python3 ../sabre/sabre_new.py \
#     -m ../sabre/traces/abr_test/pensieve_big.json \
#     -n ../sabre/traces/abr_test/network.json \
#     -a ../sabre/bb_pos.py \
#     -p starting_bitrate=3 reservoir=5 cushion=10

import os
import csv
from datetime import datetime
from typing import Any
from collections import deque

import numpy as np
import tensorflow.compat.v1 as tf  # not used by BB, but SABRE imports tf in many ABR files
tf.disable_eager_execution()

from sabre_new import Abr

try:
    from sabre_new import get_buffer_level, manifest
except Exception:
    get_buffer_level = None  # type: ignore
    manifest = None          # type: ignore

import json


# ---------- Policy/Video settings ----------
BRS = [300, 750, 1200, 1850, 2850, 4300]  # kbps
_A_DIM = len(BRS)

# (Optional) QoE parameters used only for logging (no RL here)
R_MIN = 260 # for gR to be nonzero

def g(R_kbps):
    """Log-normalized video quality function."""
    # return float(np.log(max(R_kbps, 1e-6) / R_MIN))
    return np.log(R_kbps / R_MIN)

class bb_pos(Abr):
    """
    Buffer-based ABR for SABRE:
      if buffer < reservoir -> lowest bitrate
      if buffer > reservoir + cushion -> highest bitrate
      else linearly map buffer to a bitrate index
    """

    def __init__(self, config):
        super().__init__(config)

        # User parameters from -p k=v
        self._starting_bitrate = int(config.get("starting_bitrate", 3))  # 0..5 index
        self._reservoir = float(config.get("reservoir", 5.0))  # seconds
        self._cushion   = float(config.get("cushion", 10.0))   # seconds

        # Internal trackers for logging / QoE
        self._last_quality_idx = self._starting_bitrate
        self._buffer_at_decision_s = 0.0
        self._cumulative_qoe = 0.0

        # QoE calculation
        bitrate_kbps = BRS[self._last_quality_idx]
        gR = g(bitrate_kbps)
        # Add quality term
        self._cumulative_qoe += gR

        self._last_gR = gR

        # CSV log setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, f"bb_log_{timestamp}.csv")
        self._log_writer = csv.writer(open(self._log_file, "w", newline=""))
        self._log_writer.writerow(["segment_index", "bitrate_kbps", "download_time_ms", "cumulative_qoe", "average_qoe"])

        print(f"📄 Logging to: {self._log_file}")

        # (Optional) keep short histories for debugging/inspection
        self._thr_hist = deque(maxlen=8)
        self._delay_hist = deque(maxlen=8)
        self._br_hist = deque(maxlen=8)

    # ---------------- SABRE interface ----------------

    def get_first_quality(self):
        """
        Return the initial quality index (0.._A_DIM-1).
        """
        self._last_quality_idx = int(np.clip(self._starting_bitrate, 0, _A_DIM - 1))
        # Best effort: record buffer at decision time (before first download)
        self._buffer_at_decision_s = self._safe_buffer_seconds()
        return self._last_quality_idx

    def get_quality_delay(self, segment_index: int):
        """
        Decide bitrate based on current buffer BEFORE downloading this segment.
        Return (quality_index, extra_delay_ms).
        """
        buf_s = self._safe_buffer_seconds()
        self._buffer_at_decision_s = buf_s

        if buf_s < self._reservoir:
            q = 0
        elif buf_s >= self._reservoir + self._cushion:
            q = _A_DIM - 1
        else:
            # Linear mapping into [0, _A_DIM-1]
            frac = (buf_s - self._reservoir) / max(self._cushion, 1e-6)
            q = int(np.floor(frac * (_A_DIM - 1)))
            q = int(np.clip(q, 0, _A_DIM - 1))

        self._last_quality_idx = q
        return q, 0.0  # no extra delay insertion

    def report_download(self, metrics: Any, is_replacement: bool):
        """
        SABRE calls this after a segment has finished downloading.
        metrics has fields: index, time (ms), downloaded (bytes), etc.
        """
        seg_idx  = float(metrics.index)
        delay_ms = max(float(metrics.time), 1e-3)
        bytes_dl = int(metrics.downloaded)

        # Throughput estimate in kB/ms
        thr_kB_per_ms = (bytes_dl / 1000.0) / delay_ms

        # # Rebuffer (approximate): if download time exceeds buffer available at decision time
        # rebuffer_s = max(0.0, (delay_ms / 1000.0) - self._buffer_at_decision_s)

        # QoE calculation
        bitrate_kbps = BRS[self._last_quality_idx]
        gR = g(bitrate_kbps)
        # Add quality term
        # print(self._cumulative_qoe)
        self._cumulative_qoe += gR

        # Add smoothness penalty
        smoothness_penalty = 0.0
        if self._last_gR is not None:
            smoothness_penalty = abs(gR - self._last_gR)
            # print(smoothness_penalty)
            # print(self._cumulative_qoe)
            self._cumulative_qoe -= smoothness_penalty

        self._last_gR = gR

        # Calculate average QoE
        avg_qoe = float(self._cumulative_qoe) / float(seg_idx)
        # print(self._cumulative_qoe / seg_idx)
        self._log_writer.writerow([seg_idx, bitrate_kbps, delay_ms, self._cumulative_qoe, avg_qoe])
        print(f"[Chunk {int(seg_idx)}] Bitrate={bitrate_kbps} kbps, Download Time={delay_ms} ms, QoE={self._cumulative_qoe}, AvgQoE={avg_qoe}")
        
        # Update small histories (optional)
        self._thr_hist.append(thr_kB_per_ms)
        self._delay_hist.append(delay_ms)
        self._br_hist.append(self._last_quality_idx)

    def report_delay(self, delay):
        pass

    def report_seek(self, where):
        pass

    # ---------------- Helpers ----------------

    def _safe_buffer_seconds(self) -> float:
        """Return current buffer in seconds, or 0.0 if unavailable."""
        try:
            if get_buffer_level is None:
                return 0.0
            return float(get_buffer_level()) / 1000.0
        except Exception:
            return 0.0
