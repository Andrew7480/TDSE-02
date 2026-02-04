import os
import numpy as np


def model_fn(model_dir):

    model_path = os.path.join(model_dir, "logreg_model_age_chol.npy")
    model = np.load(model_path, allow_pickle=True).item()
    return model

def predict_fn(input_data, model):
    X = np.array(input_data, dtype=float)
    X_norm = (X - model["mu"]) / model["sigma"]
    z = np.dot(X_norm, model["w"]) + model["b"]
    prob = 1 / (1 + np.exp(-z))
    return float(prob)
