# inference.py
import json
import numpy as np
import os

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "logreg_model_full.npy")
    model = np.load(model_path, allow_pickle=True).item()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        # expects a list under "inputs"
        return np.array(payload["inputs"], dtype=float)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # input_data shape: (n_features,) or (batch, n_features)
    X = np.array(input_data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    mu = model["mu"]
    sigma = model["sigma"]
    w = model["w"]
    b = model["b"]

    X_norm = (X - mu) / sigma
    z = X_norm @ w + b
    prob = 1 / (1 + np.exp(-z))
    return prob

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        # return probabilities as JSON
        result = {"probability": prediction.tolist()}
        return json.dumps(result)
    raise ValueError(f"Unsupported content type: {response_content_type}")