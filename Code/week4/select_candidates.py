# select_candidates.py
# Select candidate points by least confidence and run PGD attack at each epsilon to get empirical adversarials.
import os
import json
import torch
import numpy as np
from attacks import pgd_linf
from datetime import datetime
from torchvision import datasets, transforms
import argparse
from train_mnist_conv import MNISTConv

# epsilons scaled to MNIST [0,1]: pixel range (2/255, 5/255, 10/255, 20/255).
#2/55 tiney while 20/255 more visible
EPSILONS = [2/255., 5/255., 10/255., 20/255.]

def load_latest_pth():
    #sorted alphabetically because they should contain time stamps
    files = sorted([f for f in os.listdir("models") if f.endswith(".pth")])
    if not files:
        raise FileNotFoundError("No .pth found in models/")
    #return latest model checkpoint
    return os.path.join("models", files[-1])

def compute_margins(model, device):
    #load MNIST test dataset and wrap it in a DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
    #set to evaluation mode
    model.to(device).eval()
    #logits are the pre-softmax class scores for each batch
    logits_all = []
    labels_all = []
    #iterates over data without computing gradient to save memory
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            #collect logits and all true labels
            logits_all.append(logits)
            labels_all.append(yb.numpy())
    #combine batch info into full arrays
    logits_all = np.vstack(logits_all)
    labels_all = np.concatenate(labels_all)

    # margin = top logit- second top logit
    #large margin means confident, small means uncertain (close to decision boundary)
    top2 = np.sort(logits_all, axis=1)[:,-2:]
    margins = top2[:,-1] - top2[:,-2]
    return margins, logits_all, labels_all, testset

#generate adverserial examples using PGD
def run(num_candidates=40, restarts=5, iters=100, step_ratio=0.25):
    os.makedirs("results", exist_ok=True)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pth = load_latest_pth()

    #instantiate same model architecture and load saved weights
    model = MNISTConv()
    model.load_state_dict(torch.load(pth, map_location="cpu"))
    model.to(device).eval()         #move to device and eval mode

    #compute how confident model is on all test examples
    margins, logits_all, labels_all, testset = compute_margins(model, device)
    
    # select least confident indices aka smallest margin
    idxs = np.argsort(margins)[:num_candidates]
    candidates = []

    #for each candidated image
    for idx in idxs:
        img, label = testset[idx]
        x = img.unsqueeze(0).to(device)
        #add metadata
        info = {"index": int(idx), "label": int(label), "margins": float(margins[idx])}
        info["pgd"] = {}
        #run PGD attack for each epsilon value above
        #larger eps means stronger attack
        for eps in EPSILONS:
            # step size default
            step = max(eps * step_ratio, 1e-6)
            found, adv = pgd_linf(model, x.clone(), label, eps=eps, step=step, iters=iters, restarts=restarts, device=device)
            #if attack successfully fools model, save adverserial image flattened into a list
            if found:
                adv_cpu = adv.squeeze(0).cpu().numpy().tolist()
                info["pgd"][str(eps)] = {"found": True, "adv_flat": adv_cpu}
            #otherwise record failure
            else:
                info["pgd"][str(eps)] = {"found": False, "adv_flat": None}
        candidates.append(info)

    # save to JSON - timestamps, full list of attacks
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/candidates_{ts}.json"
    with open(out, "w") as f:
        json.dump(candidates, f, indent=2)
    print("Saved candidates to", out)
    
    #return filepath incase needed for downstream analysis
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-candidates", type=int, default=40)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    run(num_candidates=args.num_candidates, restarts=args.restarts, iters=args.iters)
