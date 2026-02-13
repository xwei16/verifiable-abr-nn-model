import numpy as np

# -----------------------------
# Pensieve input dimensions
# -----------------------------
num_samples = 100
input_shape = (6, 8)
input_dim = 6 * 8  # 48 total input features
output_dim = 6     # Pensieve outputs 6 actions

# -----------------------------
# Generate realistic samples
# -----------------------------
samples = []

for _ in range(num_samples):
    sample = np.zeros((6, 8))

    # Row 0: Past throughput (Mbps)
    sample[0, :] = np.random.uniform(0.3, 5.0, 8)

    # Row 1: Download time (seconds)
    sample[1, :] = np.random.uniform(0.5, 4.0, 8)

    # Row 2: Next chunk sizes (KB)
    sample[2, :] = np.random.uniform(200, 3000, 8)

    # Row 3: Buffer occupancy (seconds)
    sample[3, :] = np.random.uniform(0.0, 10.0, 8)

    # Row 4: Remaining chunks
    sample[4, :] = np.random.uniform(0.0, 50.0, 8)

    # Row 5: Current bitrate level
    sample[5, :] = np.random.uniform(0.0, 5.0, 8)

    samples.append(sample.flatten())

samples = np.array(samples)

# -----------------------------
# Compute per-dimension bounds
# -----------------------------
lower_bounds = samples.min(axis=0)
upper_bounds = samples.max(axis=0)

# -----------------------------
# Write ABCROWN-compatible VNNLIB
# -----------------------------
with open("spec/1.vnnlib", "w") as f:

    # Declare input variables
    for i in range(input_dim):
        f.write(f"(declare-const X_{i} Real)\n")

    # Declare output variables (optional but fine)
    for i in range(output_dim):
        f.write(f"(declare-const Y_{i} Real)\n")

    f.write("\n; Input constraints from sampled data\n")

    # IMPORTANT: one assert per constraint
    for i in range(input_dim):
        f.write(f"(assert (>= X_{i} {lower_bounds[i]:.10f}))\n")
        f.write(f"(assert (<= X_{i} {upper_bounds[i]:.10f}))\n")
    
    # Dummy output constraints
    f.write("\n; Dummy output constraints (force ABCROWN to compute bounds)\n")
    for i in range(output_dim):
        f.write(f"(assert (>= Y_{i} -1000000000.0))\n")
        f.write(f"(assert (<= Y_{i}  1000000000.0))\n")

print("✅ VNNLIB spec generated (ABCROWN compatible)")
print("   ABCROWN will compute output bounds automatically.")