import csv
import numpy as np
import json

input_file = "../spectra_original_data/testing_files/pensieve_events_pensieve_best_big_test.csv"
output_file = "pensieve_big_testing_data.csv"

def compute_qoe_2(last, curr):
    last_q = np.log(last * 4300 / 300)  # Denormalize first 
    curr_q = np.log(curr / 300)
    qoe_2 = last_q + curr_q - abs(last_q - curr_q)
    return qoe_2

with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames + ['qoe_2']
    rows = []

    for row in reader:
        last_chunk_bitrate = float(row['Last1_chunk_bitrate'])
        br = float(row['br'])
        row['qoe_2'] = compute_qoe_2(last_chunk_bitrate, br)  # Normalize for log formula

        rows.append(row)

# Write to new CSV with qoe_2 column
with open(output_file, 'w', newline='') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
