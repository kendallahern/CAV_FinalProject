# Week 4 — MNIST ConvNet Verification (Z3 + PGD + IBP)

This week focuses on deeper, higher-quality robustness analysis of a single
well-trained MNIST ConvNet. The workflow mirrors recent robust-ML literature:

1. Train a small ConvNet ("MiniNet") with ReLU activations.
2. Export the model to `.pth`, `.json` (SMT-friendly), and `.onnx` (future use).
3. Identify 8–10 critical evaluation points:
   - least confident clean samples
   - adversarially perturbed variants (PGD)
4. Use interval bound propagation (IBP) to discard impossible points cheaply.
5. Use Z3 to formally check:
   - for all x' within ||x' - x||∞ ≤ ε: predicted class = y
   - for ε in {2/255, 5/255, 10/255, 20/255}

Results are stored in:

```bash
results/
   logs/
   candidates/
   attacks/
   z3/
   z3/mini
   case_reports/
```

## Setup Instructions

```bash
cd week4
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** May need to do

```bash
python3 -m pip install -r requirements.txt
```

### Download MNIST

```bash
python download_mnist.py
```

### Train MiniNet

```bash
python train_mnist_conv.py --epochs 5 --batch-size 128
```

Outputs will be timestamped like this:

```pgsql
models/20251114_153322_MiniNet.pth
models/20251114_153322_MiniNet.json
models/20251114_153322_MiniNet.onnx
```

And logged to:

```bash
results/logs/train_log.csv
```

---

### After training...

1. Run `python3 find_candidates.py`— this creates results/candidates/points_<ts>.json and image files. Inspect these images to ensure PGD saved advs.

2. Run `python3 ibp_bounds.py` --candidates results/candidates/points_<ts>.json — this creates triage JSON. This helps skip obviously-provable or obviously-violated cases before doing SMT.

3. Run `python3 verify_cases.py` --triage results/candidates/triage_<ts>.json — by default it runs only ambiguous cases. Increase --timeout-ms for longer Z3 runs.

```bash
python3 verify_cases.py --model-json models/20251117_110717_mnist_conv.json  
```
Only ambiguous points

```bash
python3 mini_verify_cases.py --model-json models/20251117_104023_MiniNet.json
```
All triaged points

4. Run `python3 generate_case_reports.py` — produces markdown reports from results/z3/.