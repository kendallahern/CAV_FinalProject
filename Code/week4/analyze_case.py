# analyze_case.py
# Given a z3 result file or candidate, produce plots: original image, adversarial, diff heatmap,
# per-layer activation differences and ReLU flip counts.

import pickle
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
import torch
from train_mnist_conv import MNISTConv
from torchvision import datasets, transforms

def load_image_by_index(idx):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    img, label = testset[idx]
    return img.numpy(), int(label)

def compute_activations(model, x_tensor):
    """
    Returns list of (layer_name, z_numpy, a_numpy) tuples
    """
    activations = []
    x = x_tensor.clone().unsqueeze(0)
    with torch.no_grad():
        z1 = model.conv1(x); a1 = torch.relu(z1)
        z2 = model.conv2(a1); a2 = torch.relu(z2)
        a2p = torch.nn.functional.max_pool2d(a2, 2)
        flat = a2p.view(a2p.size(0), -1)
        z3 = model.fc1(flat); a3 = torch.relu(z3)
        z4 = model.fc2(a3)
    activations.append(("conv1", z1.squeeze(0).cpu().numpy(), a1.squeeze(0).cpu().numpy()))
    activations.append(("conv2", z2.squeeze(0).cpu().numpy(), a2.squeeze(0).cpu().numpy()))
    activations.append(("pool", a2p.squeeze(0).cpu().numpy(), a2p.squeeze(0).cpu().numpy()))
    activations.append(("fc1", z3.squeeze(0).cpu().numpy(), a3.squeeze(0).cpu().numpy()))
    activations.append(("fc2", z4.squeeze(0).cpu().numpy(), z4.squeeze(0).cpu().numpy()))
    return activations

def safe_save_pickle(obj, path):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def run_analysis(case_json, model_pth=None, output_dir=None):
    """
    case_json: path to the per-case file saved by verify_cases.py
    - This file should contain {"triage": ..., "verify": {..., "counterexample": [flat list]}}
    """
    with open(case_json) as f:
        res = json.load(f)

    # infer sample index from filename if possible
    base = os.path.basename(case_json)
    idx = None
    parts = base.split("_")
    for p in parts:
        if p.isdigit():
            idx = int(p); break
    if idx is None:
        # fallback: if triage info contains index
        if isinstance(res.get("triage"), dict) and "index" in res["triage"]:
            idx = int(res["triage"]["index"])
        else:
            raise ValueError("Cannot infer sample index from filename or triage block.")

    x0, label = load_image_by_index(idx)

    # load model
    if model_pth is None:
        pths = sorted([f for f in os.listdir("models") if f.endswith(".pth")])
        if not pths:
            raise FileNotFoundError("No .pth found in models/")
        model_pth = os.path.join("models", pths[-1])
    model = MNISTConv()
    model.load_state_dict(torch.load(model_pth, map_location="cpu"))
    model.eval()

    # compute original activations
    x_tensor = torch.tensor(x0, dtype=torch.float32)
    acts_orig = compute_activations(model, x_tensor)

    # process counterexample if present
    ce_flat = res.get("verify", {}).get("counterexample", None) or res.get("counterexample", None)
    adv_arr = None
    acts_adv = None
    if ce_flat:
        # sanity checks: must be flat list length 784
        if not isinstance(ce_flat, list):
            raise ValueError("Counterexample is not a list; cannot reshape.")
        if len(ce_flat) != 28*28:
            # save the raw CE for inspection, but skip analysis
            print(f"Counterexample length unexpected ({len(ce_flat)}), skipping adv analysis.")
        else:
            adv_arr = np.array(ce_flat, dtype=float).reshape(1,28,28)
            adv_tensor = torch.tensor(adv_arr, dtype=torch.float32)
            acts_adv = compute_activations(model, adv_tensor)

    # decide output dir
    if output_dir is None:
        output_dir = os.path.join("results/case_reports", f"case_{idx}")
    os.makedirs(output_dir, exist_ok=True)

    # Save images (original and adv if present)
    plt.imsave(os.path.join(output_dir, "orig.png"), x0.squeeze(), cmap="gray")
    if adv_arr is not None:
        plt.imsave(os.path.join(output_dir, "adv.png"), adv_arr.squeeze(), cmap="gray")
        diff = np.abs(adv_arr - x0)
        plt.imsave(os.path.join(output_dir, "diff.png"), diff.squeeze(), cmap="hot")

    # ReLU flips and per-layer stats
    report = {"index": idx, "label": label}
    if acts_adv:
        flips = []
        for (name_o, z_o, a_o), (name_a, z_a, a_a) in zip(acts_orig, acts_adv):
            z_o_sign = (z_o > 0).astype(int)
            z_a_sign = (z_a > 0).astype(int)
            flips_layer = int(np.sum(z_o_sign != z_a_sign))
            flips.append({"layer": name_o, "flips": int(flips_layer)})
        report["relu_flips"] = flips

    # Save activations and report with pickle (safe for heterogeneous shapes)
    safe_save_pickle(acts_orig, os.path.join(output_dir, "acts_orig.pkl"))
    if acts_adv:
        safe_save_pickle(acts_adv, os.path.join(output_dir, "acts_adv.pkl"))

    # Save JSON summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Saved case analysis to", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--pth", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    run_analysis(args.case, model_pth=args.pth, output_dir=args.out)
