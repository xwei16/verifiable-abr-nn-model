import pickle
import numpy as np

# Load the results (adjust path to your actual results file)
with open('qoe_bounds.pkl', 'rb') as f:  # or whatever your results file is
    results = pickle.load(f)

print("Available keys:", results.keys())

# The bounds you want are likely in these fields:
print("\n=== Initial CROWN Bounds ===")
if 'init_crown_bounds' in results:
    init_bounds = results['init_crown_bounds']
    print(f"Type: {type(init_bounds)}")
    print(f"Content: {init_bounds}")

print("\n=== Refined Lower Bounds ===")
if 'refined_lb' in results:
    refined_lb = results['refined_lb']
    print(f"Type: {type(refined_lb)}")
    print(f"Refined lower bounds: {refined_lb}")

# If these are tensors, convert to numpy
if hasattr(refined_lb, 'numpy'):
    refined_lb = refined_lb.numpy()
    
print("\n=== Prediction Info ===")
print(f"Prediction: {results.get('pred', 'N/A')}")
print(f"Attack margin: {results.get('attack_margin', 'N/A')}")

# More detailed extraction
print("\n=== Detailed Bound Extraction ===")
for key in ['init_crown_bounds', 'refined_lb']:
    if key in results:
        data = results[key]
        if isinstance(data, (list, tuple)):
            print(f"\n{key} (list/tuple with {len(data)} elements):")
            for i, elem in enumerate(data):
                print(f"  Element {i}: shape={getattr(elem, 'shape', 'N/A')}, type={type(elem)}")
                if hasattr(elem, 'shape') and len(elem.shape) <= 2:
                    print(f"    Values: {elem}")
        elif hasattr(data, 'shape'):
            print(f"\n{key}: shape={data.shape}")
            print(f"  Values: {data}")
        else:
            print(f"\n{key}: {data}")