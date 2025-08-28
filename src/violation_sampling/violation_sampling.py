import os
import sys
import csv
from datetime import datetime
import numpy as np
# Add sabre directory to Python path
sabre_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sabre"))
sys.path.append(sabre_dir)

from pos import pos  # Now this works

# --- Pensieve bitrate options ---
BRS = [300, 750, 1200, 1850, 2850, 4300]

# --- Number of segments ---
TOTAL_SEGMENTS = 3


# -----------------------------
# Mock download simulator
# -----------------------------
def simulate_download(bitrate_index, segment_index):
    base_size = 1 * 1024 * 1024  # 1MB base
    downloaded = base_size * (1 + 0.5 * bitrate_index)
    time_ms = 500 + 100 * bitrate_index  # e.g., 500ms to 1000ms

    class Metrics:
        def __init__(self, index, downloaded, time):
            self.index = index
            self.downloaded = downloaded
            self.time = time

    return Metrics(index=segment_index, downloaded=int(downloaded), time=time_ms)

# -----------------------------
# Initialize Pensieve ABR agent
# -----------------------------
config = {
    "starting_bitrate": 3
}
agent = pos(config)


# --- Generate timestamped filename ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"pensieve_sim_{timestamp}.csv"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
LOG_PATH = os.path.join(log_dir, log_filename)

# -----------------------------
# Setup CSV Logging
# -----------------------------
with open(LOG_PATH, "w", newline="") as logfile:
    writer = csv.writer(logfile)
    writer.writerow(["segment_index", "bitrate_kbps", "download_time_ms", "cumulative_qoe"])

    cumulative_qoe = 0.0

    for seg in range(TOTAL_SEGMENTS):
        if seg == 0:
            br_index = agent.get_first_quality()
        else:
            br_index, _ = agent.get_quality_delay(segment_index=seg)

        bitrate_kbps = BRS[br_index]
        metrics = simulate_download(br_index, seg)

        reward = bitrate_kbps  # Example placeholder QoE
        cumulative_qoe += reward

        agent.report_download(metrics, is_replacement=False)

        writer.writerow([seg, bitrate_kbps, metrics.time, cumulative_qoe])

        print(f"[Segment {seg}] Bitrate: {bitrate_kbps} kbps, Download Time: {metrics.time:.1f} ms, QoE: {cumulative_qoe:.1f}")

print(f"\n✅ Simulation complete. Logged results to: {LOG_PATH}")
