import matplotlib.pyplot as plt
import numpy as np

# ---- Raw 90% CI data extracted from your table ----

dates = [
    "2025-09-20",
    "2025-09-30",
    "2025-10-28",
    "2025-10-29",
    "2025-10-30",
    "2025-10-31",
    "2025-11-01",
    "2025-11-02",
    "2025-11-03",
    "2025-11-04",
    "Combined Model"
]

# 2-layer models (10 individual + combined)
two_layer_CIs = [
    (0.9806,0.9931,0.8151,0.8633),
    (0.9794,0.9923,0.9146,0.9674),
    (0.9882,0.9974,0.9862,1.0000),
    (0.9782,0.9915,0.9317,0.9764),
    (0.9856,0.9960,0.9855,0.9998),
    (0.9869,0.9967,0.9841,1.0000),
    (0.9664,0.9834,0.9339,0.9814),
    (0.9699,0.9859,0.9607,0.9846),
    (0.9584,0.9775,0.9933,1.0000),
    (0.9722,0.9875,0.7903,0.8494),
    # combined model
    (0.9966,0.9983,0.9846,0.9887)
]

# 3-layer models (10 individual)
three_layer_CIs = [
    (0.9831,0.9946,0.8616,0.9039),
    (0.9856,0.9960,0.9738,0.9985),
    (0.9856,0.9960,0.9781,0.9998),
    (0.9882,0.9974,0.9887,1.0000),
    (0.9843,0.9953,0.9909,1.0000),
    (0.9895,0.9980,0.9841,1.0000),
    (0.9595,0.9783,0.9397,0.9846),
    (0.9819,0.9938,0.9673,0.9888),
    (0.9607,0.9792,0.9933,1.0000),
    (0.9758,0.9899,0.8253,0.8799)
]

def plot_ci(data, title, filename):
    train_mid = [(a+b)/2 for (a,b,_,_) in data]
    test_mid  = [(a+b)/2 for (_,_,a,b) in data]

    train_err = [(b-a)/2 for (a,b,_,_) in data]
    test_err  = [(b-a)/2 for (_,_,a,b) in data]

    x = np.arange(len(data))
    width = 0.35

    plt.figure(figsize=(10, 6))
    
    plt.bar(x - width/2, 
            train_mid, 
            width, 
            yerr=train_err, 
            edgecolor="black",      # Border color
            linewidth=1.5,           # Border thickness
            label='Training Data')
    plt.bar(x + width/2, 
            test_mid, 
            width, 
            yerr=test_err, 
            edgecolor="black",      # Border color
            linewidth=1.5,           # Border thickness
            label='Testing Data')

    if len(data) <= 10:
        plt.xticks(x, dates[:-1], rotation=45, ha="right")
    else:
        plt.xticks(x, dates, rotation=45, ha="right")
    plt.ylabel("Accuracy (Midpoint of 90% CI)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
   

    # Save as SVG
    if "2-Layer" in title:
        plt.ylim(0.8, 1.02)
    else:
        plt.ylim(0.8, 1.03)
    plt.subplots_adjust(left=0.12)
    plt.savefig(filename, format='svg')
    plt.close()

# ---- Generate & Save SVGs ----
plot_ci(two_layer_CIs, "Clopper Pearson Bound – 2-Layer Models (90% CI)", "two_layer_models.svg")
plot_ci(three_layer_CIs, "Clopper Pearson Bound – 3-Layer Models (90% CI)", "three_layer_models.svg")