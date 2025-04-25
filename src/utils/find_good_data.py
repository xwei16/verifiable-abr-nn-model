import pandas as pd 
import json
import numpy as np
import argparse 

with open("qoe_spec.json", "r") as f:
    specs = json.load(f)

def match_spec(row, spec):
    """Check if a row satisfies the spec's feature bounds."""
    for feat in ['Last1_chunk_bitrate', 'Last1_buffer_size',
                 'Last1_downloadtime', 'Last2_downloadtime', 'Last3_downloadtime']:
        l = spec.get(f"{feat}_l")
        u = spec.get(f"{feat}_u")
        if not (l <= round(row.get(feat),5) <= u):
            return False
        
    if not ((row.get('br') in spec.get('cur_br')) and (row.get('qoe_2') in spec.get('qoe_2'))):
        return False
    return True

def find_good_data(df, specs):

    all_matches = pd.DataFrame()

    for spec in specs:
        # Find rows that match the spec conditions
        matched_rows = df[df.apply(lambda row: match_spec(row, spec), axis=1)]
    
    all_matches = pd.concat([all_matches, matched_rows], ignore_index=True)
    

    if not all_matches.empty:
        out_name = "filtered_good_testing_data.csv"
        all_matches.to_csv(out_name, index=False)

# get specs 
def main():
    df_test = pd.read_csv(f'pensieve_big_testing_data.csv')
    # Run evaluation
    find_good_data(df_test, specs)

if __name__ == '__main__':
    main()