# verify_local_robustness.py
"""
Main verification driver for Week 2.

Globals near the top control experiments:
    EPSILONS, MARGINS, NUM_TEST_POINTS, SELECT_BY, TIMEOUT_MS, PHASE_MODE, MODELS_DIR, RESULTS_DIR

Run as:
    python verify_local_robustness.py --model models/tiny_net_weights.json
or call run_experiments.py to run batch across models.
"""

import os
import json
import argparse
import numpy as np
import csv
from time import time
from smt_encoder import verify_local_robustness
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------- GLOBALS (edit these) -------------------
EPSILONS = [0.02, 0.05, 0.1, 0.2]
MARGINS  = [0.0, 0.1]
NUM_TEST_POINTS = 10
SELECT_BY = "least_confident"  # "least_confident" | "random" | "manual"
TIMEOUT_MS = 15000  # Z3 timeout for each solve (ms)
PHASE_MODE = "if"  # "if" or "bool"
MODELS_DIR = "models"
RESULTS_DIR = "results"
# ------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

def load_model_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def flatten_weights(data):
    keys = list(data["weights"].keys())
    layers = {}
    for k in keys:
        prefix, rest = k.split('.', 1)
        layers.setdefault(prefix, {})[rest] = np.array(data["weights"][k])
    ordered = []
    i = 1
    while True:
        key = f"fc{i}"
        if key in layers:
            ordered.append((layers[key]["weight"], layers[key]["bias"]))
            i += 1
        else:
            break
    return ordered, data.get("scaler_mean", None), data.get("scaler_scale", None)

def build_dataset_and_splits(n_samples=500):
    # must match train_variants.py settings: make_moons noise and random_state
    X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    return X, y

def select_test_points(weights_biases, scaler_mean, scaler_scale, select_by="least_confident", num_points=10, seed=0):
    # Recreate dataset split identically to training script
    X, y = build_dataset_and_splits(n_samples=500)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # If scaler_mean exists (from model), use that to normalize (consistent with training)
    if scaler_mean is not None and scaler_scale is not None:
        # override scaler to use saved values
        scaler.mean_ = np.array(scaler_mean)
        scaler.scale_ = np.array(scaler_scale)
        X_scaled = (X - scaler.mean_) / scaler.scale_

    # reproduce train/test split (same random_state)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # compute confidence margins on X_test using float forward
    def float_forward(wb, x):
        a = x.copy()
        last_z = None
        for (W, b) in wb:
            z = W.dot(a) + b
            last_z = z
            a = np.maximum(0.0, z)
        return last_z

    margins = []
    logits_list = []
    for i, xt in enumerate(X_test):
        logits = float_forward(weights_biases, xt)
        logits_list.append(logits)
        # for binary: margin = abs(logit0 - logit1)
        # for multi-class: margin = max - second_max
        sorted_idx = np.argsort(-logits)
        margin = float(logits[sorted_idx[0]] - logits[sorted_idx[1]])
        margins.append((i, margin, logits))

    if select_by == "least_confident":
        margins_sorted = sorted(margins, key=lambda x: abs(x[1]))  # ascending margin -> least confident
        selected = [m[0] for m in margins_sorted[:num_points]]
    elif select_by == "random":
        rng = np.random.RandomState(seed)
        idxs = np.arange(len(X_test))
        rng.shuffle(idxs)
        selected = list(idxs[:num_points])
    else:
        # manual - pick first num_points
        selected = list(range(min(num_points, len(X_test))))

    test_points = [X_test[i] for i in selected]
    test_labels = [int(y_test[i]) for i in selected]
    return np.array(test_points), np.array(test_labels), X_test, y_test

def run_one_model(model_path, out_csv_writer):
    print(f"Running model: {model_path}")
    data = load_model_json(model_path)
    wb, scaler_mean, scaler_scale = flatten_weights(data)

    # pick test points
    X_sel, y_sel, X_test_full, y_test_full = select_test_points(
        wb, scaler_mean, scaler_scale, select_by=SELECT_BY, num_points=NUM_TEST_POINTS)

    for idx, (x0, y_true) in enumerate(zip(X_sel, y_sel)):
        for eps in EPSILONS:
            for delta in MARGINS:
                tstart = time()
                res = verify_local_robustness(wb, x0, eps, delta=delta, phase_mode=PHASE_MODE, timeout_ms=TIMEOUT_MS)
                tend = time()
                row = {
                    "model": os.path.basename(model_path),
                    "point_idx": idx,
                    "eps": eps,
                    "delta": delta,
                    "result": res.get("result"),
                    "pred_class": res.get("pred_class"),
                    "ambiguous": res.get("ambiguous"),
                    "solve_time_s": res.get("time_s"),
                    "total_time_s": res.get("total_time_s"),
                    "counterexample": json.dumps(res.get("counterexample")) if res.get("counterexample") else ""
                }
                out_csv_writer.writerow(row)
                # flush to disk
                print(" ->", row)
    print("Finished model:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to single model json. If omitted, runs all in MODELS_DIR.")
    args = parser.parse_args()

    # CSV results file
    csv_path = os.path.join(RESULTS_DIR, "logs", "results.csv")
    header = ["model", "point_idx", "eps", "delta", "result", "pred_class", "ambiguous", "solve_time_s", "total_time_s", "counterexample"]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        # choose models
        model_paths = []
        if args.model:
            model_paths = [args.model]
        else:
            # all json in MODELS_DIR
            for fname in os.listdir(MODELS_DIR):
                if fname.endswith(".json"):
                    model_paths.append(os.path.join(MODELS_DIR, fname))
        for mp in sorted(model_paths):
            run_one_model(mp, writer)
    print("All experiments done. Results saved to", csv_path)
