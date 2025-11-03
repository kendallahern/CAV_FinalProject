# analyze_case.py
# Given a z3 result file or candidate, produce plots: original image, adversarial, diff heatmap,
# per-layer activation differences and ReLU flip counts.
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import torch
from train_mnist_conv import MNISTConv
from torchvision import datasets, transforms

def load_image_by_index(idx):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    img, label = testset[idx]
    return img.numpy(), int(label)

def compute_activations(model, x_tensor):
    # returns list of (z,a) per layer
    activations = []
    x = x_tensor.clone().unsqueeze(0)
    with torch.no_grad():
        # conv1
        z1 = model.conv1(x)
        a1 = torch.relu(z1)
        # conv2
        z2 = model.conv2(a1)
        a2 = torch.relu(z2)
        # pool
        a2p = torch.nn.functional.max_pool2d(a2, 2)
        flat = a2p.view(a2p.size(0), -1)
        z3 = model.fc1(flat)
        a3 = torch.relu(z3)
        z4 = model.fc2(a3)
    activations.append(("conv1", z1.squeeze(0).cpu().numpy(), a1.squeeze(0).cpu().numpy()))
    activations.append(("conv2", z2.squeeze(0).cpu().numpy(), a2.squeeze(0).cpu().numpy()))
    activations.append(("pool", a2p.squeeze(0).cpu().numpy(), a2p.squeeze(0).cpu().numpy()))
    activations.append(("fc1", z3.squeeze(0).cpu().numpy(), a3.squeeze(0).cpu().numpy()))
    activations.append(("fc2", z4.squeeze(0).cpu().numpy(), z4.squeeze(0).cpu().numpy()))
    return activations

def run_analysis(case_json, model_pth=None, output_dir=None):
    with open(case_json) as f:
        res = json.load(f)
    # attempt to infer index from filename
    base = os.path.basename(case_json)
    # try to parse 'case_<index>' pattern
    idx = None
    parts = base.split("_")
    for p in parts:
        if p.isdigit():
            idx = int(p); break
    if idx is None:
        raise ValueError("Cannot infer sample index from filename; pass index manually.")
    x0, label = load_image_by_index(idx)
    # load model
    if model_pth is None:
        pths = sorted([f for f in os.listdir("models") if f.endswith(".pth")])
        model_pth = os.path.join("models", pths[-1])
    model = MNISTConv()
    model.load_state_dict(torch.load(model_pth, map_location="cpu"))
    model.eval()
    # compute original activations
    x_tensor = torch.tensor(x0, dtype=torch.float32)
    acts_orig = compute_activations(model, x_tensor)
    # adversarial if present
    adv = res.get("counterexample")
    if adv:
        adv_arr = np.array(adv).reshape(1,28,28)
        adv_tensor = torch.tensor(adv_arr, dtype=torch.float32)
        acts_adv = compute_activations(model, adv_tensor)
    else:
        adv_arr = None
        acts_adv = None

    # output dir
    if output_dir is None:
        output_dir = f"results/case_reports/case_{idx}"
    os.makedirs(output_dir, exist_ok=True)
    # Save images
    plt.imsave(os.path.join(output_dir, "orig.png"), x0.squeeze(), cmap="gray")
    if adv_arr is not None:
        plt.imsave(os.path.join(output_dir, "adv.png"), adv_arr.squeeze(), cmap="gray")
        diff = np.abs(adv_arr - x0)
        plt.imsave(os.path.join(output_dir, "diff.png"), diff.squeeze(), cmap="hot")
    # ReLU flips
    report = {"index": idx, "label": label}
    flips = []
    if acts_adv:
        for (name_o, z_o, a_o), (name_a, z_a, a_a) in zip(acts_orig, acts_adv):
            # count sign changes pre-activation
            z_o_sign = (z_o > 0).astype(int)
            z_a_sign = (z_a > 0).astype(int)
            flips_layer = int(np.sum(z_o_sign != z_a_sign))
            flips.append((name_o, flips_layer))
        report["relu_flips"] = flips
    # Save activations to npy for inspection
    np.save(os.path.join(output_dir, "acts_orig.npy"), acts_orig, allow_pickle=True)
    if acts_adv:
        np.save(os.path.join(output_dir, "acts_adv.npy"), acts_adv, allow_pickle=True)
    # write json summary
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
