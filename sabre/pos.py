# python3 ../sabre/sabre.py -m ../sabre/traces/abr_test/two_chunk.json -n ../sabre/traces/abr_test/network.json -a ../sabre/pos.py

import os
from collections import deque
from typing import Any, List

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

# -----------------------------------------------------------------------------
#  Pensieve‑to‑Sabre 适配器 (pos.py)
# -----------------------------------------------------------------------------
#  * 占位符 / Softmax 名称取自用户提供的 ckpt。
#  * 状态矩阵严格按 Pensieve 训练时 6×8 排列与归一。
# -----------------------------------------------------------------------------

from sabre import Abr, AbrInput  # Sabre 基类

try:  # 运行时 sabre 已加载并定义这些符号
    from sabre import get_buffer_level, manifest
except Exception:
    get_buffer_level = None  # type: ignore
    manifest = None  # type: ignore

# ------------------------------ 模型参数 --------------------------------------
_S_INFO = 6            # state rows
_S_LEN = 8             # history window
_A_DIM = 6             # bitrate choices
_BUFFER_NORM = 10.0
_CHUNK_REMAIN_CAP = 48.0
_MODEL_CKPT = os.path.expanduser(
    "../pensieve_rl_model/pretrain_linear_reward.ckpt")

# ckpt 内 tensor 名称（已确认）
_TENSOR_IN  = "actor/InputData/X:0"
_TENSOR_OUT = "actor/FullyConnected_4/Softmax:0"

# --------------------------- Sabre ABR 实现 ----------------------------------
class pos(Abr):
    """Sabre wrapper for a pre‑trained Pensieve policy."""

    # ------------------------------------------------------------------
    #  初始化
    # ------------------------------------------------------------------
    def __init__(self, config):
        super().__init__(config)

        # 1) 创建 Session 并载入图
        self._sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                                      inter_op_parallelism_threads=1))
        saver = tf.train.import_meta_graph(_MODEL_CKPT + ".meta")
        saver.restore(self._sess, _MODEL_CKPT)
        g = tf.get_default_graph()
        self._s_in  = g.get_tensor_by_name(_TENSOR_IN)
        self._q_out = g.get_tensor_by_name(_TENSOR_OUT)

        # 2) 历史轨迹缓存
        self._last_q_hist = deque([0.0] * _S_LEN, maxlen=_S_LEN)   # row0
        self._buffer_hist = deque([0.0] * _S_LEN, maxlen=_S_LEN)   # row1 (sec/10)
        self._thrpt_hist  = deque([1.0] * _S_LEN, maxlen=_S_LEN)   # row2 (KB/ms)
        self._delay_hist  = deque([0.4] * _S_LEN, maxlen=_S_LEN)   # row3 (delay/10)
        self._next_sizes  = np.zeros(_A_DIM, dtype=np.float32)      # row4 (MB)
        self._remain_hist = deque([1.0] * _S_LEN, maxlen=_S_LEN)   # row5 (remain/48)

        self._last_quality = 0  # 上一次决策

    # ------------------------------------------------------------------
    #  Sabre 接口
    # ------------------------------------------------------------------
    def get_first_quality(self):
        return 0  # 启动码率

    def get_quality_delay(self, segment_index: int):
        """返回 (quality_index, extra_delay_ms)."""
        # 拼 (6,8) 状态矩阵
        state_mat = np.vstack([
            np.array(self._last_q_hist),                    # row0 last bitrate (0~1)
            np.array(self._buffer_hist),                    # row1 buffer /10
            np.array(self._thrpt_hist),                     # row2 throughput KB/ms
            np.array(self._delay_hist),                     # row3 delay /10
            np.pad(self._next_sizes, (0, _S_LEN - _A_DIM)), # row4 next sizes MB
            np.array(self._remain_hist),                    # row5 remain /48
        ], dtype=np.float32)

        probs = self._sess.run(self._q_out, {self._s_in: state_mat[np.newaxis, ...]})[0]
        quality = int(np.argmax(probs))
        self._last_quality = quality
        return quality, 0.0

    # ------------------------------------------------------------------
    #  下载完成回调
    # ------------------------------------------------------------------
    def report_download(self, metrics: Any, is_replacement: bool):  # type: ignore
        """metrics ≡ sabre.DownloadProgress namedtuple."""
        # -----------------------------------------------------------
        # 基本度量
        delay_ms   = max(metrics.time, 1e-3)   # 避免除零
        bytes_dl   = metrics.downloaded
        seg_idx    = metrics.index

        # throughput KB/ms
        kb_per_ms = bytes_dl / 1000.0 / delay_ms

        # buffer level (秒)
        if get_buffer_level is not None:
            buffer_ms = get_buffer_level()
            buffer_sec = buffer_ms / 1000.0
        else:
            buffer_sec = 0.0  # Fallback

        # remaining segments (保守估算)
        if manifest is not None and hasattr(manifest, 'segments'):
            remain = max(len(manifest.segments) - seg_idx - 1, 0)
        else:
            remain = 0

        # next segment sizes (Bytes → MB)
        if manifest is not None and seg_idx + 1 < len(manifest.segments):
            next_sizes_bits = manifest.segments[seg_idx + 1]
            next_sizes_mb = np.array(next_sizes_bits[:_A_DIM], dtype=np.float32) / 8.0 / 1e6
        else:
            next_sizes_mb = np.zeros(_A_DIM, dtype=np.float32)

        # -----------------------------------------------------------
        #  更新历史
        self._last_q_hist.append(float(self._last_quality) / (_A_DIM - 1))
        self._buffer_hist.append(buffer_sec / _BUFFER_NORM)
        self._thrpt_hist.append(kb_per_ms)
        self._delay_hist.append(delay_ms / 10000.0)  # /1000 /10
        self._remain_hist.append(min(remain, _CHUNK_REMAIN_CAP) / _CHUNK_REMAIN_CAP)
        self._next_sizes = next_sizes_mb

    # 其它回调保留空实现
    def report_delay(self, delay):
        pass

    def report_seek(self, where):
        pass
