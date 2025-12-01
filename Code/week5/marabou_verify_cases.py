import os
import json
import argparse
from datetime import datetime
from torchvision import datasets, transforms
import numpy as np

from marabou_verify_point import marabou_verify_point
from analyze_case import run_analysis


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def run(triage_json, onnx_path, model_pth, timeout=60, delta=0.0, analyze=True, eps_zero=False, debug=False, single_branch=False):

    ensure_dir("results/marabou")

    # Load triage JSON
    with open(triage_json) as f:
        triage_list = json.load(f)

    # Load MNIST testset
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    all_results = []

    for entry in triage_list:
        idx = entry["index"]

        for eps_str, info in entry["per_eps"].items():
            eps = float(eps_str)
            print(f"\n[Marabou] idx={idx}   eps={eps}")

            img, _ = testset[idx]
            x0 = img.numpy()

            # Progress callback for Marabou
            def progress_callback(stats):
                nodes = stats.get('nodesExplored', '?')
                t = stats.get('time', 0.0)
                print(f"[Progress] Nodes explored: {nodes} | Time elapsed: {t:.2f}s", end='\r')

            # Call Marabou verification
            out = marabou_verify_point(
                onnx_path,
                model_pth,
                x0,
                eps=0.0 if eps_zero else eps,
                delta=delta,
                timeout=timeout,
                debug=debug,
                single_branch=single_branch,
                #progress_callback=progress_callback
            )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"results/marabou/case_{idx}_eps_{int(eps*255)}_{ts}.json"

            # Save per-case result
            with open(fname, "w") as f:
                json.dump({"triage": info, "verify": out}, f, indent=2)

            # Run post-analysis
            if analyze:
                try:
                    run_analysis(
                        fname,
                        model_pth=model_pth,
                        output_dir=f"results/case_reports_marabou/case_{idx}"
                    )
                except Exception as e:
                    print("Analysis failed:", e)

            # Add to summary
            all_results.append({
                "index": idx,
                "eps": eps,
                "status": info.get("status", "unknown"),
                "marabou": out
            })

    # Save summary
    summary_path = f"results/marabou/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--model-pth", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--single-branch", action="store_true")
    parser.add_argument("--eps-zero", action="store_true")

    args = parser.parse_args()

    run(
        triage_json=args.triage,
        onnx_path=args.onnx,
        model_pth=args.model_pth,
        timeout=args.timeout,
        delta=args.delta,
        analyze=args.analyze,
        eps_zero=args.eps_zero,
        debug=args.debug,
        single_branch=args.single_branch
    )
