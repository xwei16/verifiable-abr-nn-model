# python inspect_bounds.py pensieve_bounds_CROWN-Optimized.npz
import numpy as np
import argparse

def inspect_bounds_file(npz_file):
    """Inspect the contents of a bounds file."""
    
    data = np.load(npz_file)
    
    print("=" * 80)
    print(f"BOUNDS FILE: {npz_file}")
    print("=" * 80)
    
    print("\n📊 OUTPUT BOUNDS (Logits):")
    print("-" * 80)
    lb = data['lower'][0]
    ub = data['upper'][0]
    
    print(f"{'Action':<10} {'Lower Bound':>15} {'Upper Bound':>15} {'Range':>12}")
    print("-" * 80)
    for i in range(len(lb)):
        print(f"{i:<10} {lb[i]:>15.2f} {ub[i]:>15.2f} {ub[i]-lb[i]:>12.2f}")
    
    print("\n📥 INPUT BOUNDS:")
    print("-" * 80)
    
    # Previous bitrate
    if 'prev_bitrate_lower' in data:
        print(f"Previous Bitrate: [{data['prev_bitrate_lower']:.2f}, {data['prev_bitrate_upper']:.2f}] Mbps")
        print(f"Previous Bitrate Index: [{data['prev_bitrate_index_lower']}, {data['prev_bitrate_index_upper']}]")
    
    # Throughput
    if 'throughput_lower' in data:
        tp_lb = data['throughput_lower']
        tp_ub = data['throughput_upper']
        print(f"Throughput: [{tp_lb.min():.2f}, {tp_ub.max():.2f}] Mbps")
    
    # Buffer
    if 'buffer_lower' in data:
        buf_lb = data['buffer_lower']
        buf_ub = data['buffer_upper']
        print(f"Buffer: [{buf_lb.min():.2f}, {buf_ub.max():.2f}] seconds")
    
    # Download time
    if 'download_time_lower' in data:
        dt_lb = data['download_time_lower']
        dt_ub = data['download_time_upper']
        print(f"Download Time: [{dt_lb.min():.2f}, {dt_ub.max():.2f}] seconds")
    
    # Chunk sizes
    if 'chunk_sizes_lower' in data:
        cs_lb = data['chunk_sizes_lower']
        cs_ub = data['chunk_sizes_upper']
        print(f"Chunk Sizes: [{cs_lb.min():.2f}, {cs_ub.max():.2f}] KB")
    
    # Remaining chunks
    if 'remaining_lower' in data:
        rem_lb = data['remaining_lower']
        rem_ub = data['remaining_upper']
        print(f"Remaining Chunks: [{rem_lb.min():.2f}, {rem_ub.max():.2f}]")
    
    print("\n" + "=" * 80)
    
    # List all keys
    print("\n📋 All keys in file:")
    for key in data.keys():
        print(f"  - {key}: shape={data[key].shape if hasattr(data[key], 'shape') else 'scalar'}")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect bounds file')
    parser.add_argument('npz_file', help='Path to .npz bounds file')
    args = parser.parse_args()
    
    inspect_bounds_file(args.npz_file)