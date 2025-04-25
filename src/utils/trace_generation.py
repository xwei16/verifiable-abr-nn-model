import pandas as pd
import numpy as np
import argparse
import json
import re
from pathlib import Path

def estimate_chunk_duration():
    """
    Estimate the chunk duration from the Pensieve testing data.
    The function reads a CSV file containing bitrate and chunk size information,
    computes the chunk duration based on the selected bitrate, and returns the
    estimated chunk duration in seconds and milliseconds.
    """
    # 0.  Read the CSV file
    # df = pd.read_csv("pensieve_testing_data.csv")
    df = pd.read_csv("pensieve_big_testing_data.csv")

    # 1.  Find every column that stores a per‑representation size
    chunk_cols = [c for c in df.columns if c.startswith("chunksize")]
    if not chunk_cols:
        raise ValueError("No chunksize* columns in the file!")

    # 2.  Build a mapping  <bitrate value → its chunksize column>.
    #     If you **know** the ladder (e.g. 300→chunksize1, 750→chunksize2, …),
    #     just write that dict explicitly.  Otherwise derive it automatically:
    #
    #     (a)  Sort the unique `br` values            → ascending ladder order
    #     (b)  Sort the `chunksize` columns by *mean* → smallest to largest
    #
    unique_brs   = sorted(df["br"].unique())                 # e.g. [300, 750, …]
    by_mean_size = sorted(chunk_cols,
                        key=lambda c: df[c].mean())        # e.g. ['chunksize1', 'chunksize2', …]
    br2col = dict(zip(unique_brs, by_mean_size))

    # 3.  For each row take the *size* that corresponds to the bitrate that was
    #     actually chosen and compute that chunk’s duration.
    sizes_bits   = []
    bitrates_bps = []
    for _, row in df.iterrows():
        col         = br2col[row["br"]]      # the right chunksize column
        size_bits   = row[col] * 8 * 1_000_000   # MB → bits (adapt multiplier if KB/B)
        bitrate_bps = row["br"] * 1000            # kb/s → b/s
        sizes_bits.append(size_bits)
        bitrates_bps.append(bitrate_bps)

    durations_s = np.array(sizes_bits) / np.array(bitrates_bps)

    # 4.  Robust aggregate: take the *median* (or trimmed mean) to knock out noise
    segment_duration_s  = float(np.median(durations_s))
    segment_duration_ms = round(segment_duration_s * 1000)

    print(f"Estimated chunk duration ≈ {segment_duration_s:.3f} s "
        f"({segment_duration_ms} ms)")


def chunk_size_calculation():
    df = pd.read_csv("pensieve_big_testing_data.csv")
    output_file = "video_traces_helper.json"

    # every column named chunksize1, chunksize2, …
    chunk_cols = [c for c in df.columns if re.fullmatch(r"chunksize\d+", c)]
    if not chunk_cols:
        raise SystemExit("ERROR: no chunksize* columns found in the CSV.")

    # convert sizes to bits according to declared unit
    factor = 8_000_000 # MB → bits
    bits_df = (df[chunk_cols] * factor).round().astype(int)

    # list‑of‑lists
    segment_sizes_bits = bits_df.values.tolist()

    Path(output_file).write_text(json.dumps(segment_sizes_bits, indent=2))
    print(
        f"Wrote {output_file}  "
        f"({len(segment_sizes_bits)} segments, {len(chunk_cols)} reps)."
    )

    chunk_cols           = ["chunksize1", "chunksize2", "chunksize3",
                        "chunksize4", "chunksize5", "chunksize6"]
    bits_per_segment     = df[chunk_cols] * 8_000_000           # MB → bits
    segment_sizes_bits   = bits_per_segment.round().astype(int) # nice ints
    segment_sizes_bits   = segment_sizes_bits.values.tolist()   # list of lists


CSV_FILE        = Path("pensieve_big_testing_data.csv")          # input trace
SIZES_OUT       = Path("segment_sizes_bits.json")
BANDWIDTH_OUT   = Path("bandwidth_kbps.json")
SIZE_UNIT       = "MB"        # MB  |  KB  |  B

def to_bits(series, unit):
    """Convert sizes to bits according to declared unit."""
    factor = {"MB": 8_000_000, "KB": 8_000, "B": 8}[unit.upper()]
    return series * factor

def main():
    if not CSV_FILE.exists():
        raise SystemExit(f"{CSV_FILE} not found.")

    df = pd.read_csv(CSV_FILE)

    # 1  find all chunksize columns
    chunk_cols = [c for c in df.columns if re.fullmatch(r"chunksize\d+", c)]
    if not chunk_cols:
        raise SystemExit("No columns like 'chunksize1', 'chunksize2', … found.")

    # 2  auto‑map  br (kbps)  → matching chunksize column
    unique_brs   = sorted(df["br"].unique())
    cols_by_size = sorted(chunk_cols, key=lambda c: df[c].mean())
    if len(unique_brs) != len(cols_by_size):
        raise SystemExit("Cannot infer bitrate ladder: "
                         "number of distinct 'br' values ≠ chunksize columns.")
    br2col = dict(zip(unique_brs, cols_by_size))   # e.g. 300 → "chunksize1"

    # 3  segment_sizes_bits  (convert every chunksize column to bits)
    sizes_bits_df = to_bits(df[cols_by_size], SIZE_UNIT)
    segment_sizes_bits = sizes_bits_df.round().astype(int).values.tolist()
    SIZES_OUT.write_text(json.dumps(segment_sizes_bits, indent=2))
    print(f"Wrote {SIZES_OUT}  "
          f"({len(segment_sizes_bits)} segments × {len(cols_by_size)} reps).")

    # 4  bandwidth_kbps  =  size_bits / download_time / 1000
    bw_samples = []
    for _, row in df.iterrows():
        col          = br2col[row["br"]]
        size_bits    = to_bits(row[col], SIZE_UNIT)
        dl_time      = row["Last1_downloadtime"]
        bw_kbps      = float("nan") if dl_time <= 0 else size_bits / dl_time / 1000
        bw_samples.append(round(bw_kbps))
    BANDWIDTH_OUT.write_text(json.dumps(bw_samples, indent=2))
    print(f"Wrote {BANDWIDTH_OUT}  ({len(bw_samples)} samples).")


main()