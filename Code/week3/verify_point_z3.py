# verify_point_z3.py
import argparse
import numpy as np
from z3_encoder import verify_point
import json, os

def load_candidate_json(path, idx):
    with open(path) as f:
        c = json.load(f)
    return c[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="model json file")
    parser.add_argument("--candidate", type=str, default=None, help="candidate json (from select_candidates)")
    parser.add_argument("--pt-index", type=int, default=0)
    parser.add_argument("--eps", type=float, default=5/255.)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=30000)
    args = parser.parse_args()

    if args.model is None:
        # pick latest
        files = sorted([f for f in os.listdir("models") if f.endswith("_mnist_conv.json")])
        args.model = os.path.join("models", files[-1])

    if args.candidate is None:
        # fallback: load latest candidates
        cand_files = sorted([f for f in os.listdir("results") if f.startswith("candidates_") and f.endswith(".json")])
        if not cand_files:
            raise FileNotFoundError("No candidates json found. Run select_candidates.py first.")
        args.candidate = os.path.join("results", cand_files[-1])

    with open(args.candidate) as f:
        cands = json.load(f)
    c = cands[args.pt_index]
    # reconstruct x0 tensor from PIL-like flattened info in PGD? select_candidates saved img not flattened; but we have index -> load MNIST test image
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    img, label = testset[c["index"]]
    x0 = img.numpy()  # shape (1,28,28)
    out = verify_point(args.model, x0, eps=args.eps, delta=args.delta, timeout=args.timeout)
    # save result
    os.makedirs("results/z3", exist_ok=True)
    fname = f"results/z3/case_{c['index']}_eps_{args.eps:.5f}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved z3 result to", fname)
    print(out)
