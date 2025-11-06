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
    
    # Compute downloading_time = (ack_time - sent_time)/1e6 - rtt_ms
    downloading_time = (ack_time - sent_time) / 1e3 - rtt_ms  # microseconds
    
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

    # Convert to arrays
    bandwidths = np.array(bandwidths)
    
    # Merge every 9 rows into one
    n_complete_groups = len(bandwidths) // 9
    merged_rows = []
    
    for i in range(n_complete_groups):
        start_idx = i * 9
        end_idx = start_idx + 9
        
        # Create a single row with:
        # - First 8 rows' chunk_size and downloading_time
        # - 9th row's bandwidth, delay, chunk_size, and downloading_time
        row = {}
        
        # Add first 8 rows' data
        for j in range(8):
            idx = start_idx + j
            row[f"chunk_size_{j+1}"] = size[idx] / 1e6  # mega-bytes
            row[f"downloading_time_{j+1}"] = downloading_time[idx] / 1e6  # seconds
        
        # Add 9th row's data
        idx_9th = start_idx + 8
        row["bandwidth"] = bandwidths[idx_9th]  # Mbps
        row["delay"] = rtt_ms[idx_9th] / 1e6  # seconds
        row["chunk_size_9"] = size[idx_9th] / 1e6  # mega-bytes
        row["downloading_time_9"] = downloading_time[idx_9th] / 1e6  # seconds
        
        merged_rows.append(row)
    
    # Create DataFrame
    df_out = pd.DataFrame(merged_rows)
    
    # Reorder columns for clarity
    column_order = []
    for j in range(1, 9):
        column_order.append(f"chunk_size_{j}")
        column_order.append(f"downloading_time_{j}")
    column_order.extend(["bandwidth", "delay", "chunk_size_9", "downloading_time_9"])
    df_out = df_out[column_order]
    
    # Save to CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out.to_csv(output_file, index=False)

    print(f"✅ Saved {len(df_out)} rows to {output_file}")
    print(f"   (merged from {n_complete_groups * 9} original rows)")

if __name__ == "__main__":
    process_puffer(
        "../../puffer_data/video_sent_2025-09-20T11_2025-09-21T11.csv",
        "../../puffer_data/video_acked_2025-09-20T11_2025-09-21T11.csv",
        "output/network_data_training_1000_new.txt",
        group_size=100,
        max_rows=1000
    )

    process_puffer(
        "../../puffer_data/video_sent_2025-09-30T11_2025-10-01T11.csv",
        "../../puffer_data/video_acked_2025-09-30T11_2025-10-01T11.csv",
        "output/network_data_testing_1000_new.txt",
        group_size=100,
        max_rows=1000
    )