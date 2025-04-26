import pandas as pd
import numpy as np

input_file = 'log_sim_mpc_combined_norway.csv'
output_file = 'log_sim_mpc_cleaned_norway.csv'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    lines = infile.readlines()
    header = lines[0]
    expected_columns = len(header.strip().split(','))
    outfile.write(header)  # keep the header

    for line_num, line in enumerate(lines[1:], start=2):
        columns = line.strip().split(',')
        if len(columns) == expected_columns:
            outfile.write(line)
        else:
            print(f"Skipping line {line_num}: has {len(columns)} columns, expected {expected_columns}")