# mini_verify_cases.py
# Phase 4 adapted for MiniNet: run Z3 on all triaged candidates
# Saves results to results/z3_mini/
#
import os
import json
import argparse
from mini_z3_encoder import mini_verify_point
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

def load_latest_model_pth():
    files = sorted([f for f in os.listdir("models") if f.endswith(".pth")])
    if not files:
        raise FileNotFoundError("No .pth found in models/")
    return os.path.join("models", files[-1])

def run(triage_json=None, model_json=None, model_pth=None, timeout_ms=60000, delta=0.0, analyze=True):
    ensure_dir("results/z3_mini")
    if triage_json is None:
        cand_files = sorted([f for f in os.listdir("results/candidates") 
                             if f.startswith("triage_") and f.endswith(".json")])
        if not cand_files:
            raise FileNotFoundError("No triage json found. Run ibp_bounds.py first.")
        triage_json = os.path.join("results/candidates", cand_files[-1])
    
    with open(triage_json) as f:
        triage = json.load(f)

    if model_json is None:
        model_json = load_latest_model_json()
    if model_pth is None:
        model_pth = load_latest_model_pth()

    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    results = []
    for entry in triage:
        idx = entry["index"]
        label = entry["label"]
        for eps_str, info in entry["per_eps"].items():
            eps = float(eps_str)
            # verify all cases, not just ambiguous
            print(f"Verifying idx={idx} eps={eps} ...")
            img, _ = testset[idx]
            x0 = img.numpy()
            out = mini_verify_point(model_json, x0, eps, delta=delta, timeout=timeout_ms)

            # save per-case json
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"results/z3_mini/case_mini_{idx}_eps_{int(round(eps*255))}_{ts}.json"
            with open(fname, "w") as f:
                json.dump({"triage": info, "verify": out}, f, indent=2)
            print("Saved z3 result to", fname)

            if analyze:
                try:
                    run_analysis(fname, model_pth=model_pth, output_dir=f"results/case_reports_mini/case_{idx}")
                except Exception as e:
                    print("Analysis failed for", fname, ":", e)

            results.append({"index": idx, "eps": eps, "status": info.get("status","unknown"), "z3_result": out})

    # summary
    summary_file = f"results/z3_mini/summary_mini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved summary to", summary_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage", type=str, default=None)
    parser.add_argument("--model-json", type=str, default=None)
    parser.add_argument("--model-pth", type=str, default=None)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    args = parser.parse_args()
    run(triage_json=args.triage, model_json=args.model_json, model_pth=args.model_pth,
        timeout_ms=args.timeout_ms, delta=args.delta, analyze=args.analyze)
