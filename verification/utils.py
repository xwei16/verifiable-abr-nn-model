from pathlib import Path
import pandas as pd
"""
Load data from all CSV files in a directory and combine them.

Parameters:
-----------
dir_path : str
    Directory path containing CSV files

Returns:
--------
X : numpy array
    Features (all columns except downloading_time_9)
y : numpy array
    Target variable (downloading_time_9)
"""
def load_network_data(dir_path, nrows=None):
    # Get all CSV files in the directory
    dir_path = Path(dir_path)
    csv_files = sorted(dir_path.glob('*.csv'))

    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {dir_path}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load and combine all files
    dfs = []
    y = []
    for csv_file in csv_files:
        print(f"Loading: {csv_file.name} with {nrows} rows")
        df = pd.read_csv(csv_file, nrows=nrows)
        y_df = df['downloading_time_9'].values
        y.append(y_df)
        df = df.drop(columns=['downloading_time_9']).values
        dfs.append(df)
        
    
    # Concatenate all dataframes
    # df = pd.concat(dfs, ignore_index=True)
    # print(f"Total rows loaded: {len(df)}")
    
    # Extract features and target
    # X = df.drop(columns=['downloading_time_9']).values
    
    
    return dfs, y