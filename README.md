# CAV Final Project: Robustness Verification of MNIST CNNs

### Author: Kendall Ahern

This repository contains a 4-week project exploring **formal verification and robustness analysis** of convolutional neural networks (CNNs) trained on MNIST. The project focuses on understanding how small perturbations to inputs can affect network predictions.

---

## Project Goals

- Train CNNs on MNIST and export weights in multiple formats (`.pth`, `.json`, `.onnx`).
- Use **SMT solvers (Z3)** to formally verify network robustness.
- Identify **robust**, **non-robust**, and **ambiguous cases** using interval bound propagation (IBP) and counterexamples.
- Analyze ambiguous cases with **visualizations of activations, ReLU flips, and adversarial inputs**.

---

## Networks

- **MNISTConv**: baseline convolutional network for verification experiments.
- **MiniNet**: custom CNN with two convolutional layers and two fully connected layers; optimized for integration with Z3 verification.

---

## Highlights

- **Automatic Z3 verification** of inputs within small epsilon-bounded perturbations.
- **Per-case analysis and visualization**, including heatmaps, activation differences, and counterexample images.
- **Flexible framework**: supports multiple network architectures and can be extended to other datasets.

---

## Repository Structure (high-level)

- `week1–week4/`: code and experiments for each week.
- `models/`: saved models in `.pth`, `.json`, `.onnx`.
- `results/`: candidate triage, Z3 outputs, and case reports.
- `papers/`: relevant literature and notes.

---

## Usage Summary

1. **Train a network** (e.g., MiniNet):

```bash
python3 mini_train_mnist_conv.py --epochs 5
```

2. Run Z3 verification:

```bash
python3 mini_verify_cases.py --model-json models/<MiniNet_model>.json
```

3. Analyze results:

```bash
python3 analyze_case.py --case results/z3/case_<idx>_<timestamp>.json
```

### Notes:

- Experiments were conducted on MNIST for clarity and efficiency.
- The full report summarizing findings, insights, and conclusions is maintained separately.
- This project demonstrates how formal verification and conservative bounds can provide guarantees about CNN behavior under input perturbations.