# find_candidates.py
# Phase 2: select 4 least-confident clean images and run PGD on each epsilon
#
# Output:
#   results/candidates/points.json
#   results/candidates/images/   (original and adv pngs)
#
import os
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from Code.week4.mini_train_mnist_conv import MiniNet
from attacks import pgd_linf
from datetime import datetime
import argparse
from PIL import Image

# Epsilons (MNIST pixel scale)
EPSILONS = [2/255., 5/255., 10/255., 20/255.]

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def find_latest_pth():
    files = sorted([f for f in os.listdir("models") if f.endswith(".pth")])
    if not files:
        raise FileNotFoundError("No .pth found in models/")
    return os.path.join("models", files[-1])

def compute_margins(model, device):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
    model.to(device).eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            logits_all.append(logits)
            labels_all.append(yb.numpy())
    logits_all = np.vstack(logits_all)
    labels_all = np.concatenate(labels_all)
    # margin = top - second top
    top2 = np.sort(logits_all, axis=1)[:,-2:]
    margins = top2[:,-1] - top2[:,-2]
    return margins, logits_all, labels_all, testset

def save_img_tensor(tensor, path):
    # tensor shape (1,28,28) or (28,28)
    arr = tensor.squeeze().clip(0,1) * 255.0
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    im.save(path)

def run(num_clean=4, restarts=5, iters=100, step_ratio=0.25):
    ensure_dir("results")
    ensure_dir("results/candidates")
    ensure_dir("results/candidates/images")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    pth = find_latest_pth()
    model = MiniNet()
    state = torch.load(pth, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    margins, logits_all, labels_all, testset = compute_margins(model, device)
    # choose least confident
    idxs = np.argsort(margins)[:num_clean]

    candidates = []
    for idx in idxs:
        img, label = testset[idx]
        x = img.unsqueeze(0).to(device)
        entry = {
            "index": int(idx),
            "label": int(label),
            "margin": float(margins[idx]),
            "orig_image_path": None,
            "pgd": {}
        }
        # save original image png
        orig_path = f"results/candidates/images/orig_{idx}.png"
        save_img_tensor(img.numpy(), orig_path)
        entry["orig_image_path"] = orig_path

        for eps in EPSILONS:
            step = max(eps * step_ratio, 1e-6)
            found, adv = pgd_linf(model, x.clone(), label, eps=eps, step=step, iters=iters, restarts=restarts, device=device)
            key = f"{eps:.6f}"
            if found:
                adv_np = adv.squeeze(0).cpu().numpy()
                adv_path = f"results/candidates/images/adv_{idx}_eps_{int(round(eps*255))}.png"
                save_img_tensor(adv_np, adv_path)
                entry["pgd"][key] = {"found": True, "adv_image_path": adv_path}
            else:
                entry["pgd"][key] = {"found": False, "adv_image_path": None}
        candidates.append(entry)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/candidates/points_{ts}.json"
    with open(out, "w") as f:
        json.dump(candidates, f, indent=2)
    print("Saved candidates to", out)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clean", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    run(num_clean=args.num_clean, restarts=args.restarts, iters=args.iters)
