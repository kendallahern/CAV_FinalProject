# Week 5 — Robustness Verification with Marabou (MiniNet)

Week 5 extends Week 4 by replacing Z3-based SMT encodings with the
Marabou verifier, which supports convolution, ReLU, and max-pooling natively.

This significantly improves scalability for CNNs like MiniNet.

## Workflow

### 1. Export MiniNet to ONNX

python3 convert_pytorch_to_onnx.py
--model-pth models/20251117_104023_MiniNet.pth
--out models/20251117_104023_MiniNet.onnx

shell
Copy code

### 2. Run Marabou verification for triaged MNIST cases

python3 marabou_verify_cases.py
--triage results/candidates/triage_20251117_105846.json
--onnx models/20251117_104023_MiniNet.onnx
--model-pth models/20251117_104023_MiniNet.pth

markdown
Copy code

### 3. Results

- JSON results in `results/marabou/`
- Case reports in `results/case_reports_marabou/`

## Notes

- Verification property: ∃ x′ in L∞ ball s.t. logits[x′] ≠ logits[x].
- Marabou supports disjunctions and ReLU natively → faster and more accurate than Z3.
- The pipeline is fully compatible with Week 4's triage and case analysis.


```bash
python3 marabou_verify_cases.py \
    --triage results/candidates/triage_20251117_105846.json \
    --onnx models/20251117_104023_MiniNet.onnx \
    --model-pth models/20251117_104023_MiniNet.pth
```

```bash
python3 marabou_verify_cases.py \
    --triage results/candidates/triage_20251117_105846.json \
    --onnx models/20251117_104023_MiniNet.onnx \
    --model-pth models/20251117_104023_MiniNet.pth \
    --eps-zero \
    --timeout 300 \
    --delta 1e-6
```

## For Marabou...

1. Create Docker file. Make sure Docker Destop is open and running. 
2. Build dockerfile

```bash
docker build -t marabou-linux .
```
3. Run the container with project folder mounted

```bash
docker run -it --rm \
    -v $(pwd):/app \
    --name marabou-test \
    marabou-linux
```

4. Activate .venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision numpy matplotlib scikit-learn z3-solver
```

5. Run inside container

```bash
python3 marabou_verify_cases.py \
    --triage results/candidates/triage_20251117_105846.json \
    --onnx models/20251117_104023_MiniNet.onnx \
    --model-pth models/20251117_104023_MiniNet.pth \
    --eps-zero \
    --timeout 300
```

```bash
/app/verify_cases.py     --triage results/candidates/triage_20251117_105846.json     --onnx models/20251117_104023_MiniNet.onnx     --model-pth models/20251117_104023_MiniNet.pth     --eps-zero     --timeout 300     --delta 1e-6     --debug
```