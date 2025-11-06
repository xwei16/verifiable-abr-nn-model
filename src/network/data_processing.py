import pandas as pd
import numpy as np
import os

def process_puffer(sent_file, acked_file, output_file, group_size=100, max_rows=1000000):
    # Load CSVs
    sent = pd.read_csv(sent_file, nrows=max_rows)
    acked = pd.read_csv(acked_file, nrows=max_rows)
    
    # Merge on unique identifiers
    merged = pd.merge(
        sent,
        acked,
        on=["video_ts"],
        suffixes=("_sent", "_acked")
    )
    
    # Extract times and fields
    sent_time = merged["time_sent"].values    # ns
    ack_time = merged["time_acked"].values    # ns
    rtt_ms = merged["rtt"].values             # micro-seconds
    size = merged["size"].values              # bytes
    # print(ack_time[0])
    # print(sent_time[0])
    # print(rtt_ms[0])
    # Compute downloading_time = (ack_time - sent_time)/1e6 - rtt_ms
    downloading_time = (ack_time - sent_time) / 1e3 - rtt_ms  # microseconds
    # print(downloading_time)
    # Filter valid rows
    valid = downloading_time > 0
    merged = merged[valid].reset_index(drop=True)
    downloading_time = downloading_time[valid]
    size = size[valid]
    rtt_ms = rtt_ms[valid]

    # Compute throughput (Mbit/s)
    throughput = size / downloading_time * 8 / 1000

    # Group rows into batches of group_size
    bandwidths = []
    for i in range(0, len(throughput), group_size):
        group_tp = throughput[i:i+group_size]
        if len(group_tp) == 0:
            continue
        bw = np.max(group_tp)  # max throughput in group
        bandwidths.extend([bw] * len(group_tp))  # assign group bw

    # # Align lengths
    # min_len = min(len(bandwidths), len(size), len(rtt_ms), len(downloading_time))
    # bandwidths = np.array(bandwidths[:min_len])
    # size = size[:min_len]
    # rtt_ms = rtt_ms[:min_len]
    # downloading_time = downloading_time[:min_len]

    # Save to text file
    df_out = pd.DataFrame({
        "bandwidth": bandwidths, #Mbps
        "delay": rtt_ms/1e6, #s
        "chunk_size": size/1e6, #mega-bytes
        "downloading_time": downloading_time/1e6 #s
    })
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out.to_csv(output_file, index=False)

    print(f"✅ Saved {len(df_out)} rows to {output_file}")

if __name__ == "__main__":
    process_puffer(
        "../../puffer_data/video_sent_2025-09-20T11_2025-09-21T11.csv",
        "../../puffer_data/video_acked_2025-09-20T11_2025-09-21T11.csv",
        "output/network_data_training_1000.txt",
        group_size=100,
        max_rows=1000
    )

    process_puffer(
        "../../puffer_data/video_sent_2025-09-30T11_2025-10-01T11.csv",
        "../../puffer_data/video_acked_2025-09-30T11_2025-10-01T11.csv",
        "output/network_data_testing_1000.txt",
        group_size=100,
        max_rows=1000
    )
