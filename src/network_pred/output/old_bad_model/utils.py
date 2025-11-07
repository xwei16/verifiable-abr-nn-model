import math

def pac_sample_complexity(delta=0.01, epsilon=0.01, k=3):
    """
    Calculate sample complexity for PAC guarantee with a linear model.
    
    Args:
        delta (float): Confidence parameter (failure prob).
        epsilon (float): Accuracy tolerance.
        k (int): Number of features (excluding bias).
        
    Returns:
        m (float): Required number of samples.

    # Example usage:
    m = pac_sample_complexity(delta=0.1, epsilon=0.1, k=3)

    # Assumptions: PAC learning often assumes that the data distribution is known and that the hypothesis space contains the true function (h is close to f).
    """
    d = k + 1  # pseudo-dimension for linear model with bias
    m = (d + math.log(1/delta)) / (epsilon ** 2)
    print(f"With δ={delta} and ε={epsilon}, for a linear predictor with {k} features "
        f"(pseudo-dimension d={d}), you need at least m ≈ {math.ceil(m)} i.i.d. samples "
        f"to guarantee that, with probability at least 1 - δ, the expected loss of your "
        f"learned predictor is within ε of the best linear predictor.")
    return m

m = pac_sample_complexity(delta=0.01, epsilon=0.01, k=3)


