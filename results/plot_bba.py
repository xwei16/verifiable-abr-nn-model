import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Parameters
# -------------------------------
bitrates = np.array([300, 750, 1200, 1850, 2850, 4300])  # kbps
reservoir = 5.0   # seconds
cushion = 10.0    # seconds
buffer_sizes = np.linspace(0, 30, 300)  # 0-30s buffer

# Ensure output folder exists
os.makedirs("plot", exist_ok=True)

# -------------------------------
# 1) BBA-style Mapping (Step Function)
# -------------------------------
chosen_bitrates_bba = []
for b in buffer_sizes:
    if b < reservoir:
        br = bitrates[0]
    elif b >= reservoir + cushion:
        br = bitrates[-1]
    else:
        frac = (b - reservoir) / max(cushion, 1e-6)
        idx = int(np.floor(frac * (len(bitrates) - 1)))
        br = bitrates[idx]
    chosen_bitrates_bba.append(br)

plt.figure(figsize=(8, 5))
plt.step(buffer_sizes, chosen_bitrates_bba, where='post', linewidth=2)
plt.axvline(reservoir, color='red', linestyle='--', label='Reservoir')
plt.axvline(reservoir + cushion, color='green', linestyle='--', label='Reservoir + Cushion')
plt.xlabel("Buffer Size (seconds)")
plt.ylabel("Chosen Bitrate (kbps)")
plt.title("BBA Mapping: Buffer Size → Bitrate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot/buffer_vs_bitrate_bba.png", dpi=300)
plt.close()

# -------------------------------
# 2) BOLA-style Mapping (Smooth Curve)
# -------------------------------
# For BOLA, we use a utility function U = log(bitrate)
utilities = np.log(bitrates / bitrates[0])  # log utility normalized
V = 5  # Lyapunov tuning parameter (adjust to make curve smoother or steeper)

chosen_bitrates_bola = []
for b in buffer_sizes:
    scores = utilities - (V / (b + 1e-6))  # Lyapunov score: maximize utility - penalty
    idx = np.argmax(scores)
    chosen_bitrates_bola.append(bitrates[idx])

plt.figure(figsize=(8, 5))
plt.plot(buffer_sizes, chosen_bitrates_bola, linewidth=2)
plt.xlabel("Buffer Size (seconds)")
plt.ylabel("Chosen Bitrate (kbps)")
plt.title("BOLA Mapping: Buffer Size → Bitrate")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot/buffer_vs_bitrate_bola.png", dpi=300)
plt.close()

print("✅ Saved plots:")
print(" - plot/buffer_vs_bitrate_bba.png")
print(" - plot/buffer_vs_bitrate_bola.png")

