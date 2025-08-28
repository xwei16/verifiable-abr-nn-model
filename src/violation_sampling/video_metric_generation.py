import csv
import json

# --- Configuration ---
SEGMENT_DURATION_MS = 3000
BITRATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]  # Pensieve BRS
N_BITRATES = len(BITRATES_KBPS)
N_SEGMENTS = 200  # Change this if needed

# --- Input and Output Paths ---
csv_path   = "../../data/pensieve_big_testing_data.csv"
output_json = "../../sabre/traces/abr_test/pensieve_big.json"

# --- Output Container ---
segment_sizes_bits = []

# --- Read and Convert CSV Rows ---
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for i, row in enumerate(reader):
        if i >= N_SEGMENTS:
            break

        # Collect all chunksizes for this row
        segment_row = []
        for j in range(1, N_BITRATES + 1):  # chunksize1 to chunksize6
            size_mb = float(row.get(f"chunksize{j}", 0.0))
            size_bits = int(size_mb * 8 * 1e6)  # MB → bits
            segment_row.append(size_bits)

        segment_sizes_bits.append(segment_row)

# --- Final Output JSON ---
output = {
    "segment_duration_ms": SEGMENT_DURATION_MS,
    "bitrates_kbps": BITRATES_KBPS,
    "segment_sizes_bits": segment_sizes_bits
}

# --- Write JSON File ---
with open(output_json, "w") as f:
    json.dump(output, f, indent=4)

print(f"Wrote {len(segment_sizes_bits)} segments to '{output_json}'")
