import argparse
import numpy as np

def main(args):
    data = np.load(args.npz)

    print(f"[✓] Loaded: {args.npz}")
    print("Keys:", list(data.keys()))

    if "lower" not in data or "upper" not in data:
        raise ValueError("NPZ must contain 'lower' and 'upper' arrays")

    lower = data["lower"]
    upper = data["upper"]

    print("\n=== Output Bounds ===")
    print("Lower bounds:")
    print(lower)

    print("\nUpper bounds:")
    print(upper)

    print("\n=== Per-action bounds ===")
    for i in range(lower.shape[-1]):
        lb = lower.flatten()[i]
        ub = upper.flatten()[i]
        print(f"Action {i}: [{lb:.6f}, {ub:.6f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True,
                        help="Path to bounds npz file")
    args = parser.parse_args()
    main(args)