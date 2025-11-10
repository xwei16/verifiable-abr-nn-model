import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')  # Updated to use v0_8 style
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create output directory for plots
output_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(output_dir, exist_ok=True)

# Read the data
df = pd.read_csv('useful_results/rand_pgd_successful_attack_spec0.csv')

# Convert numeric columns to float
numeric_columns = ['qoe_2', 'br', 'Last1_throughput', 'Last1_downloadtime']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Process data in groups of three
original_rows = []
attacked_rows = []

# Iterate through the dataframe in groups of three
for i in range(0, len(df), 3):
    if i + 1 < len(df):  # Ensure we have at least two rows
        original_rows.append(df.iloc[i])
        attacked_rows.append(df.iloc[i + 1])
        # Skip the gap line (i + 2)

# Convert to dataframes
original_df = pd.DataFrame(original_rows)
attacked_df = pd.DataFrame(attacked_rows)

# Reset indices
original_df = original_df.reset_index(drop=True)
attacked_df = attacked_df.reset_index(drop=True)

print("Original data shape:", original_df.shape)
print("Attacked data shape:", attacked_df.shape)

# 1. QoE Comparison Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=original_df['qoe_2'].dropna(), label='Original', color='blue', alpha=0.5)
sns.kdeplot(data=attacked_df['qoe_2'].dropna(), label='Attacked', color='red', alpha=0.5)
plt.title('Distribution of QoE Scores: Original vs Attacked')
plt.xlabel('QoE Score')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'qoe_distribution.png'))
plt.close()

# 2. QoE Change Distribution
plt.figure(figsize=(10, 6))
qoe_changes = original_df['qoe_2'].dropna() - attacked_df['qoe_2'].dropna()
sns.histplot(qoe_changes, bins=30, kde=True)
plt.axvline(x=qoe_changes.mean(), color='r', linestyle='--', label=f'Mean Change: {qoe_changes.mean():.2f}')
plt.title('Distribution of QoE Changes')
plt.xlabel('QoE Reduction (Original - Attacked)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'qoe_changes.png'))
plt.close()

# 3. Deviation of metrics
metrics = ['Last1_downloadtime', 'br', 'qoe_2']
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes = axes.ravel()

for idx, metric in enumerate(metrics):
    data = pd.DataFrame({
        'Original': original_df[metric].dropna(),
        'Attacked': attacked_df[metric].dropna()
    }).melt()
    sns.boxplot(data=data, x='variable', y='value', ax=axes[idx])
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].set_xlabel('')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'deviation_of_metrics.png'))
plt.close()

# 4. Proportional Deviation Analysis
plt.figure(figsize=(12, 6))
metrics = ['Last1_downloadtime', 'br', 'qoe_2']
deviations = []

for metric in metrics:
    # Calculate proportional deviation: (attacked - original) / original
    prop_dev = (attacked_df[metric].dropna() - original_df[metric].dropna()) / original_df[metric].dropna()
    deviations.append(prop_dev)

# Create boxplot of proportional deviations
plt.boxplot(deviations, labels=metrics)
plt.axhline(y=0, color='r', linestyle='--', label='No Change')
plt.title('Proportional Deviation of Metrics')
plt.ylabel('Proportional Change (Attacked - Original) / Original')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'proportional_deviations.png'))
plt.close()

print("Plots have been generated and saved in the 'plots' directory.")
print(f"Average QoE reduction: {qoe_changes.mean():.2f}")
print(f"Average original bitrate: {original_df['br'].dropna().mean():.0f} kbps")
print(f"Average attacked bitrate: {attacked_df['br'].dropna().mean():.0f} kbps")
