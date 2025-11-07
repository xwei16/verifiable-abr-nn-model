# python3 pac.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import math

import pickle

from tqdm import trange

from statsmodels.stats.proportion import proportion_confint

# ----------------------------
# 1) Load real data
# ----------------------------
def load_network_data(file_path):
    """
    Load network data from a CSV/text file with columns:
    bandwidth,delay,chunk_size,downloading_time
    """
    df = pd.read_csv(file_path)
    X = df[['bandwidth', 'delay', 'chunk_size']].values
    y = df['downloading_time'].values
    return X, y

# ----------------------------
# 2) PAC sample complexity calculator
# ----------------------------
def pac_sample_complexity(delta, epsilon, k):
    d = k + 1  # pseudo-dimension for linear model with bias
    return math.ceil((d + math.log(1/delta)) / (epsilon ** 2))

# ----------------------------
# 3) Train + evaluate model
# ----------------------------
def train_pac_linear(file_path, delta=0.1, epsilon=0.1, alpha=1.0):
    '''
    We perform regularized linear regression

    alpha value	    Effect on model
        0	        Ordinary least squares (no regularization).
        0.01 - 1	Mild regularization — reduces overfitting, good default range.
        > 10	    Strong regularization — model shrinks toward zero (can underfit).
    '''
    X, y = load_network_data(file_path)
    k = X.shape[1]

    m_required = pac_sample_complexity(delta, epsilon, k)
    print(f"PAC sample complexity (δ={delta}, ε={epsilon}): need ≥ {m_required} samples")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # model = Ridge(alpha=alpha)
    # Try polynomial features up to degree 2
    # This lets the linear model approximate nonlinear functions like division/multiplication (important for networking)
    model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("ridge", Ridge(alpha=0.1))
])
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_loss = mean_squared_error(y_train, y_pred_train)
    test_loss = mean_squared_error(y_test, y_pred_test)

    # print(f"Training MSE: {train_loss:.4f}")
    # print(f"Test MSE:     {test_loss:.4f}")
    # print(f"Generalization gap: {abs(test_loss - train_loss):.4f}")

    if len(X_train) >= m_required:
        print(f"✅ Enough samples: with probability ≥ {1 - delta}, "
              f"expected loss is within ε={epsilon} of the best linear predictor.")
    else:
        print(f"⚠️ Not enough samples: collect more data to meet PAC guarantee.")

    return model, (y_test, y_pred_test)



# ----------------------------
# 3) Train + evaluate small neural network
# ----------------------------
def train_nn(file_path, eps_rel=0.05, alpha=0.05):
    X_train, y_train = load_network_data(file_path)

    # Small neural network (2 hidden layers, modest size)
    model = MLPRegressor(hidden_layer_sizes=(32, 16),
                         activation='relu',
                         solver='adam',
                         learning_rate_init=0.001,
                         max_iter=1,
                         random_state=0)

    n_epochs = 500
    for epoch in trange(n_epochs, desc="Training NN"):
        model.fit(X_train, y_train)   # runs one epoch at a time

    # # Predictions
    # y_pred_test = model.predict(X_test)

    # # MSE
    # test_loss = mean_squared_error(y_test, y_pred_test)
    # print(f"Test MSE: {test_loss:.4f}")

    return model

# ----------------------------
# 5) Run with file
# ----------------------------
if __name__ == "__main__":
    # model, (y_test, y_pred_test) = train_pac_linear(
    #     "output/network_data.txt", delta=0.01, epsilon=0.1, alpha=0.1
    # )

    # # Clopper–Pearson evaluation on test set
    # evaluate_clopper_pearson(y_test, y_pred_test, eps_tolerance=0.1, alpha=0.1)

    model = train_nn("output/network_data_training.txt") 
    
    # Save model to file
    with open("output/nn_network_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved to output/nn_network_model.pkl")