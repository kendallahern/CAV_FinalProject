# generate_case_reports.py
# Phase 5: generate markdown reports for each z3 result JSON
# Each report contains:
#  - original image (embedded path)
#  - PGD adv image if exists
#  - Z3 counterexample (if sat)
#  - short numeric summary
#
import os
import json
import argparse
from datetime import datetime
from PIL import Image
import numpy as np

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def find_z3_results(z3_dirs=["results/z3", "results/z3_mini"]):
    """Find all JSON files in the given z3 directories"""
    all_files = []
    for z3_dir in z3_dirs:
        if os.path.exists(z3_dir):
            files = sorted([os.path.join(z3_dir, f) for f in os.listdir(z3_dir) if f.endswith(".json")])
            all_files.extend(files)
    return sorted(all_files)

def write_md_for_case(z3_json_path, out_dir="results/generated_case_reports"):
    with open(z3_json_path) as f:
        data = json.load(f)
    triage = data.get("triage", {})
    verify = data.get("verify", {})

    # parse index from filename
    base = os.path.basename(z3_json_path)
    parts = base.split("_")
    idx = None
    for p in parts:
        if p.isdigit():
            idx = int(p)
            break

    # build report dir
    ensure_dir(out_dir)
    report_name = os.path.splitext(base)[0] + ".md"
    report_path = os.path.join(out_dir, report_name)

    # original image path
    orig_path = triage.get("orig_image_path") if triage else None
    if orig_path is None and idx is not None:
        candidate_img = f"results/candidates/images/orig_{idx}.png"
        if os.path.exists(candidate_img):
            orig_path = candidate_img

    # pgd adv path
    adv_path = None
    if triage and "pgd" in triage:
        for k, v in triage.get("pgd", {}).items():
            if v.get("found"):
                adv_path = v.get("adv_image_path")
                break

    # assemble summary content
    lines = []
    lines.append(f"# Case report — {base}")
    lines.append("")
    if orig_path:
        lines.append(f"![original]({orig_path})")
        lines.append("")
    if adv_path:
        lines.append(f"![pgd_adv]({adv_path})")
        lines.append("")
    lines.append("## Z3 verification result")
    lines.append("")
    lines.append(f"- result: `{verify.get('result')}`")
    if verify.get("time_s") is not None:
        lines.append(f"- solver time (s): {verify.get('time_s')}")
    if verify.get("counterexample"):
        lines.append("- counterexample (first 28 values shown):")
        ce = verify.get("counterexample")
        lines.append("```\n" + str(ce[:28]) + "\n```")
    lines.append("")
    lines.append("## Triage info (IBP bounds)")
    lines.append("")
    if triage:
        lines.append("```json")
        lines.append(json.dumps(triage, indent=2))
        lines.append("```")
    lines.append("")

    # save md
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print("Wrote report:", report_path)
    return report_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z3dirs", type=str, nargs="+", default=["results/z3", "results/z3_mini"])
    parser.add_argument("--outdir", type=str, default="results/generated_case_reports")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    z3_files = find_z3_results(args.z3dirs)
    if not z3_files:
        print("No z3 results found in", args.z3dirs)
    for f in z3_files:
        try:
            write_md_for_case(f, out_dir=args.outdir)
        except Exception as e:
            print("Failed to write report for", f, ":", e)
