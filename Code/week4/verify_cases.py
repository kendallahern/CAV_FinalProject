# verify_cases.py
# Phase 4: For triaged candidates, run Z3 on ambiguous cases (or run on all if desired).
# Saves results to results/z3/
#
import os
import json
import argparse
from z3_encoder import verify_point  # expects function verify_point(model_json_path, x0, eps, delta, timeout)
from datetime import datetime
from torchvision import datasets, transforms
import numpy as np
from analyze_case import run_analysis    # optional: analyze and save visuals

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def load_latest_model_json():
    files = sorted([f for f in os.listdir("models") if f.endswith(".json")])
    if not files:
        raise FileNotFoundError("No model json found.")
    return os.path.join("models", files[-1])

def run(triage_json=None, model_json=None, only_ambiguous=True, timeout_ms=60000000, delta=0.0, analyze=True):
    ensure_dir("results/z3")
    if triage_json is None:
        cand_files = sorted([f for f in os.listdir("results/candidates") if f.startswith("triage_") and f.endswith(".json")])
        if not cand_files:
            raise FileNotFoundError("No triage json found. Run ibp_bounds.py first.")
        triage_json = os.path.join("results/candidates", cand_files[-1])
    with open(triage_json) as f:
        triage = json.load(f)

    if model_json is None:
        model_json = load_latest_model_json()

    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    results = []
    for entry in triage:
        idx = entry["index"]
        label = entry["label"]
        for eps_str, info in entry["per_eps"].items():
            eps = float(eps_str)
            status = info["status"]
            if only_ambiguous and status != "ambiguous":
                print(f"Skipping idx={idx} eps={eps} status={status}")
                continue
            # load image
            img, _ = testset[idx]
            x0 = img.numpy()
            print(f"Verifying idx={idx} eps={eps} ...")
            out = verify_point(model_json, x0, eps, delta=delta, timeout=timeout_ms)
            # save per-case json
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"results/z3/case_{idx}_eps_{int(round(eps*255))}_{ts}.json"
            with open(fname, "w") as f:
                json.dump({"triage": info, "verify": out}, f, indent=2)
            print("Saved z3 result to", fname)
            if analyze:
                try:
                    run_analysis(fname)
                except Exception as e:
                    print("Analysis failed for", fname, ":", e)
            results.append({"index": idx, "eps": eps, "status": status, "z3_result": out})
    # summary
    summary_file = f"results/z3/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved summary to", summary_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage", type=str, default=None)
    parser.add_argument("--model-json", type=str, default=None)
    parser.add_argument("--only-ambiguous", action="store_true", default=True)
    parser.add_argument("--timeout-ms", type=int, default=60000000)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    args = parser.parse_args()
    run(triage_json=args.triage, model_json=args.model_json,
        only_ambiguous=args.only_ambiguous, timeout_ms=args.timeout_ms,
        delta=args.delta, analyze=args.analyze)
