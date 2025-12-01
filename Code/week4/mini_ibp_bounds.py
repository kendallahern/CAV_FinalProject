# ibp_bounds.py
# Phase 3: compute IBP (interval bounds) for each candidate and epsilon,
# then label as provably_safe / provably_unsafe / ambiguous.
#
# Input: --candidates results/candidates/points_*.json
# Output: results/candidates/triage_<timestamp>.json
#
import os
import json
import numpy as np
import argparse
from Code.week4.mini_train_mnist_conv import MiniNet
from Code.week4.mini_train_mnist_conv import export_json as dummy_export  # not used; reading model json
from datetime import datetime
from torchvision import datasets, transforms
import torch

# EPSILONS consistent with find_candidates
EPSILONS = [2/255., 5/255., 10/255., 20/255.]

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def load_model_json(json_path):
    with open(json_path) as f:
        return json.load(f)

def find_latest_json():
    files = sorted([f for f in os.listdir("models") if f.endswith(".json")])
    if not files:
        raise FileNotFoundError("No model json in models/")
    return os.path.join("models", files[-1])

# --- IBP helpers (adapted for MiniNet structure used in train_mnist_conv.py)
def conv_ibp(prev_lb, prev_ub, W, b, stride=1, padding=0):
    # prev_* shape: (C,H,W)
    C_out, C_in, kh, kw = W.shape
    H, Ww = prev_lb.shape[1], prev_lb.shape[2]
    out_h = (H + 2*padding - kh)//stride + 1
    out_w = (Ww + 2*padding - kw)//stride + 1
    z_lb = np.zeros((C_out, out_h, out_w))
    z_ub = np.zeros((C_out, out_h, out_w))
    if padding > 0:
        prev_lb_p = np.pad(prev_lb, ((0,0),(padding,padding),(padding,padding)), constant_values=0.0)
        prev_ub_p = np.pad(prev_ub, ((0,0),(padding,padding),(padding,padding)), constant_values=0.0)
    else:
        prev_lb_p, prev_ub_p = prev_lb, prev_ub
    for co in range(C_out):
        for i in range(out_h):
            for j in range(out_w):
                cl, cu = 0.0, 0.0
                for ci in range(C_in):
                    for ki in range(kh):
                        for kj in range(kw):
                            w = W[co,ci,ki,kj]
                            a = prev_lb_p[ci, i+ki, j+kj]
                            bb = prev_ub_p[ci, i+ki, j+kj]
                            if w >= 0:
                                cl += w * a
                                cu += w * bb
                            else:
                                cl += w * bb
                                cu += w * a
                z_lb[co,i,j] = cl + b[co]
                z_ub[co,i,j] = cu + b[co]
    a_lb = np.maximum(0.0, z_lb)
    a_ub = np.maximum(0.0, z_ub)
    return z_lb, z_ub, a_lb, a_ub

def linear_ibp(prev_lb, prev_ub, W, b):
    out_dim = W.shape[0]
    z_lb = np.zeros(out_dim)
    z_ub = np.zeros(out_dim)
    for i in range(out_dim):
        cl, cu = 0.0, 0.0
        for j in range(W.shape[1]):
            w = W[i,j]
            if w >= 0:
                cl += w * prev_lb[j]
                cu += w * prev_ub[j]
            else:
                cl += w * prev_ub[j]
                cu += w * prev_lb[j]
        z_lb[i] = cl + b[i]
        z_ub[i] = cu + b[i]
    a_lb = np.maximum(0.0, z_lb)
    a_ub = np.maximum(0.0, z_ub)
    return z_lb, z_ub, a_lb, a_ub

def ibp_for_model_json(mjson, x0, eps):
    # mjson: dict as exported by train_mnist_conv.export_json
    # x0: numpy (1,28,28)
    # returns a dict with output z_lb/z_ub (logits)
    # Build input bounds
    lb = np.clip(x0 - eps, 0.0, 1.0)
    ub = np.clip(x0 + eps, 0.0, 1.0)
    # conv1
    conv1 = np.array(next(l for l in mjson["layers"] if l["type"]=="conv" ))
    # The json we exported in train_mnist_conv uses a list of layers; convs appear in order.
    # Let's parse sequentially
    layers = mjson["layers"]
    # conv1: layers[0]; conv2: layers[1]; fc1: layers[2]; fc2: layers[3]
    conv1_w = np.array(layers[0]["W"])
    conv1_b = np.array(layers[0]["b"])
    z1_lb, z1_ub, a1_lb, a1_ub = conv_ibp(lb, ub, conv1_w, conv1_b, stride=1, padding=0)
    # conv2
    conv2_w = np.array(layers[1]["W"])
    conv2_b = np.array(layers[1]["b"])
    z2_lb, z2_ub, a2_lb, a2_ub = conv_ibp(a1_lb, a1_ub, conv2_w, conv2_b, stride=1, padding=0)
    # pool 2x2 -> shrink: use min/max pooling on bounds
    C, H, Ww = a2_lb.shape
    # we expect shape -> after convs (no padding) H=24? depends on kernel/padding config from train script.
    # To avoid mismatches, compute pool dims
    pool_h = H // 2
    pool_w = Ww // 2
    lb_pool = np.zeros((C, pool_h, pool_w))
    ub_pool = np.zeros((C, pool_h, pool_w))
    for c in range(C):
        for i in range(pool_h):
            for j in range(pool_w):
                patch_lb = a2_lb[c, i*2:(i*2+2), j*2:(j*2+2)]
                patch_ub = a2_ub[c, i*2:(i*2+2), j*2:(j*2+2)]
                lb_pool[c,i,j] = patch_lb.min()
                ub_pool[c,i,j] = patch_ub.max()
    flat_lb = lb_pool.reshape(-1)
    flat_ub = ub_pool.reshape(-1)
    # fc1
    fc1_w = np.array(layers[2]["W"])
    fc1_b = np.array(layers[2]["b"])
    z3_lb, z3_ub, a3_lb, a3_ub = linear_ibp(flat_lb, flat_ub, fc1_w, fc1_b)
    # fc2 (logits)
    fc2_w = np.array(layers[3]["W"])
    fc2_b = np.array(layers[3]["b"])
    z4_lb, z4_ub, a4_lb, a4_ub = linear_ibp(a3_lb, a3_ub, fc2_w, fc2_b)
    return {"z_lb": z4_lb, "z_ub": z4_ub}

def triage(candidates_json, model_json_path, delta=0.0):
    with open(candidates_json) as f:
        cands = json.load(f)
    mjson = load_model_json(model_json_path)
    results = []
    for c in cands:
        idx = c["index"]
        # load image from dataset
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
        img, label = testset[idx]
        x0 = img.numpy()
        per_eps = {}
        for eps in EPSILONS:
            bounds = ibp_for_model_json(mjson, x0, eps)
            z_lb = bounds["z_lb"]
            z_ub = bounds["z_ub"]
            true_class = int(label)
            # condition provably safe: for all other classes, z_lb[true] >= z_ub[other] + delta
            safe = True
            for j in range(len(z_lb)):
                if j == true_class: continue
                if not (z_lb[true_class] >= z_ub[j] + delta):
                    safe = False
                    break
            if safe:
                label_status = "provably_safe"
            else:
                # provably unsafe: exists other where z_lb[other] >= z_ub[true] + delta
                unsafe = False
                for j in range(len(z_lb)):
                    if j == true_class: continue
                    if (z_lb[j] >= z_ub[true_class] + delta):
                        unsafe = True
                        break
                label_status = "provably_unsafe" if unsafe else "ambiguous"
            per_eps[f"{eps:.6f}"] = {
                "status": label_status,
                "z_lb": z_lb.tolist(),
                "z_ub": z_ub.tolist()
            }
        results.append({"index": idx, "label": int(label), "margin": c.get("margin", None), "per_eps": per_eps})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/candidates/triage_{ts}.json"
    ensure_dir("results/candidates")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved triage results to", out)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument("--model-json", type=str, default=None)
    parser.add_argument("--delta", type=float, default=0.0)
    args = parser.parse_args()
    if args.candidates is None:
        # pick latest
        cand_files = sorted([f for f in os.listdir("results/candidates") if f.endswith(".json")])
        if not cand_files:
            raise FileNotFoundError("No candidate json found. Run find_candidates.py first.")
        args.candidates = os.path.join("results/candidates", cand_files[-1])
    if args.model_json is None:
        jsons = sorted([f for f in os.listdir("models") if f.endswith(".json")])
        if not jsons:
            raise FileNotFoundError("No model json found. Run train_mnist_conv.py first.")
        args.model_json = os.path.join("models", jsons[-1])
    triage(args.candidates, args.model_json, delta=args.delta)
