# verify_improved.py
"""
Improved verifier with interval bound propagation (IBP) and selective phase encoding.
- Load in model weights from a JSOn file
- compute interval bounds forwards through the network (tight-ish IBP)
- ReLUs will need to be marked as wither active, inactive, or ambiguous
    - inactive/activs nodes need to encode simplified linear constraints
    - ambiguous nodes use one of phase mode optons below.

phase-mode options:
    if      : use If(z>=0, z, 0) for ambiguous ReLU nodes (simple)
    bool    : introduce Bool 'phase' variables for ambiguous nodes and add implication constraints

Usage:
    python3 verify_improved.py

    python3 verify_improved.py --test-index 2 --eps 0.1

    python3 verify_improved.py \
        --model models/tiny_net_weights.json \
        --pt-model models/tiny_net.pth \
        --test-index 0 \
        --num-test-points 10 \
        --eps 0.05 \
        --phase-mode if \
        --timeout None

    NOTES: if the flag is not specified, than what is on the line abouve is hte default values

"""
import argparse, json, time, glob
from z3 import *
import numpy as np
import torch

from train_variants import *
from fractions import Fraction

# ---------- utilities ----------
def load_model_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def flatten_weights(data):
    # gather ordered layers: fc1, fc2, fc3... ignore naming mismatch across models
    keys = list(data["weights"].keys())
    # simple heuristic: group by prefix before '.'
    layers = {}
    for k in keys:
        prefix, rest = k.split('.', 1)
        layers.setdefault(prefix, {})[rest] = np.array(data["weights"][k])
    # order by fc1, fc2, fc3
    ordered = []
    i = 1
    while True:
        key = f"fc{i}"
        if key in layers:
            w = layers[key].get("weight")
            b = layers[key].get("bias")
            ordered.append((w, b))
            i += 1
        else:
            break
    return ordered, data.get("scaler_mean", None), data.get("scaler_scale", None)

# ---------- Interval Bound Propagation (IBP) ----------
def ibp_bounds(weights_biases, x0, eps):
    """
    Definition: IBP is a n AI technique for verifying and training deep neural netwroks to be robust against adverserial attacks. It works by propagating the upper and lower bounds of each neuron's activation through the networks layers ot determine the range of possible output values, given an input that is within a certain perturbation radius. The point is that this should guarantee that a network will not change its prediction for a input that has been slightly altered 

    Compute conservative lower/upper pre-activation bounds for each layer using interval arithmetic.
    weights_biases: list of (W,b) pairs (numpy arrays)
    x0: numpy array (input)
    eps: scalar (L_inf)
    Returns:
        bounds: list of dicts for each layer with keys 'z_l' list of (l,u) and 'a_l' list of (l,u)

    NOTES: 
        - new bounds can be derived from previous layer's bounds and the layer's weights and biases
        - for a given input region, like the ball of radius eps., IBP computs the minimum and maximum possible outputs for each neuron during a forward pass
        - the during back propagation (training) the gradients are computed using the bounds which trains the network to be more robust
        - each loss is a combination of some standard loss (cross entropy example) and an adverserial loss (based on the worst-case output from the verification step)
        - use conservative interval propogation (from DeepPoly Paper) - compute z_lb and z_ub for each neuron - which is a standard preprocessing methos used to prune ReLU split cases
    """
    # Input bounds
    lb = x0 - eps
    ub = x0 + eps
    bounds = []
    prev_a_lb, prev_a_ub = lb, ub

    for layer_idx, (W, b) in enumerate(weights_biases):
        # z = W * a_prev + b
        # For interval: z_lb = sum_{j} min_wj, z_ub = sum_{j} max_wj
        z_lb = np.zeros(W.shape[0])
        z_ub = np.zeros(W.shape[0])
        for i in range(W.shape[0]):
            #compute contribution per input dimension
            contrib_l = 0.0
            contrib_u = 0.0
            for j in range(W.shape[1]):
                w = W[i, j]
                if w >= 0:
                    contrib_l += w * prev_a_lb[j]
                    contrib_u += w * prev_a_ub[j]
                else:
                    contrib_l += w * prev_a_ub[j]
                    contrib_u += w * prev_a_lb[j]
            z_lb[i] = contrib_l + b[i]
            z_ub[i] = contrib_u + b[i]
        # a = ReLU(z) so bounds:
        a_lb = np.maximum(0.0, z_lb)
        a_ub = np.maximum(0.0, z_ub)
        bounds.append({
            "z_lb": z_lb,
            "z_ub": z_ub,
            "a_lb": a_lb,
            "a_ub": a_ub
        })
        prev_a_lb, prev_a_ub = a_lb, a_ub
    return bounds

#functions to hep with z3 encodings
def add_affine_constraints(solver, inputs, W, b, layer_name):
    z_vars = []
    for i in range(W.shape[0]):
        expr = Sum([RealVal(float(W[i, j])) * inputs[j] for j in range(W.shape[1])]) + RealVal(float(b[i]))
        z_i = Real(f"{layer_name}_z{i}")
        solver.add(z_i == expr)
        z_vars.append(z_i)
    return z_vars

def add_relu_if_constraints(solver, z_vars, a_vars, layer_name):
    # a_i == If(z_i >= 0, z_i, 0)
    for i, z_i in enumerate(z_vars):
        solver.add(a_vars[i] == If(z_i >= 0, z_i, RealVal(0)))

def add_relu_bool_constraints(solver, z_vars, a_vars, layer_name):
    # use Bool phases: phase -> a==z, not(phase) -> a==0 and z<=0
    for i, z_i in enumerate(z_vars):
        phase = Bool(f"{layer_name}_phase{i}")
        solver.add(Implies(phase, a_vars[i] == z_i))
        solver.add(Implies(phase, z_i >= 0))
        solver.add(Implies(Not(phase), a_vars[i] == RealVal(0)))
        solver.add(Implies(Not(phase), z_i <= 0))
        # Note: don't add explicit disjunction; solver must decide phase

def select_least_confident_points(model, X, y, num_points=10):
    """
    Selects points with lowest classification confidence based on
    logit margin = |logit_class0 − logit_class1| (binary case)
    
    Returns:
        selected_X (np.ndarray): shape [num_points, 2]
        selected_y (np.ndarray): shape [num_points]
        selected_indices (list[int])
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy()
    
    # Compute confidence margins: smaller => less robust
    margins = np.abs(logits[:, 0] - logits[:, 1])
    
    # Sort indices by ascending margin → least confident first
    sorted_indices = np.argsort(margins)
    selected_indices = sorted_indices[:num_points]
    
    selected_X = X[selected_indices]
    selected_y = y[selected_indices]
    
    return selected_X, selected_y, selected_indices.tolist()

# ----------------------------------------
# Main VERIFY Function
# ----------------------------------------
def verify_model(json_path, test_point, eps=0.05, phase_mode="if", timeout=None):
    data = load_model_json(json_path)
    wb, scaler_mean, scaler_scale = flatten_weights(data)
    # model inputs expect normalized features if scaler provided
    x0 = np.array(test_point, dtype=float)
    if scaler_mean is not None:
        # assume x0 is already normalized (we'll use normalized test points); report to user otherwise
        pass

    # compute IBP bounds
    bounds = ibp_bounds(wb, x0, eps)

    # Build Z3 model
    s = Solver()
    if timeout:
        s.set("timeout", timeout)

    # Input variables
    n_in = x0.shape[0]
    x_vars = [Real(f"x_{i}") for i in range(n_in)]
    for i in range(n_in):
        s.add(x_vars[i] >= x0[i] - eps)
        s.add(x_vars[i] <= x0[i] + eps)

    # Forward encode layers using IBP to simplify ReLUs
    current_inputs = x_vars
    layer_count = len(wb)
    z_vars_all = []
    a_vars_all = []
    for layer_idx, ((W, b), layer_bounds) in enumerate(zip(wb, bounds)):
        layer_name = f"L{layer_idx+1}"
        # add affine
        z_vars = add_affine_constraints(s, current_inputs, W, b, layer_name)
        a_vars = [Real(f"{layer_name}_a{i}") for i in range(len(z_vars))]

        # Use IBP bounds to detect stable ReLUs
        z_lb = layer_bounds["z_lb"]
        z_ub = layer_bounds["z_ub"]
        stable_active = (z_lb >= 0)
        stable_inactive = (z_ub <= 0)
        ambiguous = ~(stable_active | stable_inactive)

        # for stable nodes, add direct linear constraints
        for i in range(len(z_vars)):
            if stable_active[i]:
                # a == z
                s.add(a_vars[i] == z_vars[i])
            elif stable_inactive[i]:
                # a == 0
                s.add(a_vars[i] == RealVal(0))
                # also add z <= 0 to tighten
                s.add(z_vars[i] <= RealVal(float(z_ub[i])))
            else:
                # ambiguous: choose encoding mode
                if phase_mode == "if":
                    s.add(a_vars[i] == If(z_vars[i] >= 0, z_vars[i], RealVal(0)))
                elif phase_mode == "bool":
                    phase = Bool(f"{layer_name}_phase{i}")
                    s.add(Implies(phase, a_vars[i] == z_vars[i]))
                    s.add(Implies(phase, z_vars[i] >= 0))
                    s.add(Implies(Not(phase), a_vars[i] == RealVal(0)))
                    s.add(Implies(Not(phase), z_vars[i] <= 0))
                else:
                    # fallback
                    s.add(a_vars[i] == If(z_vars[i] >= 0, z_vars[i], RealVal(0)))

        #continue forward
        current_inputs = a_vars
        z_vars_all.append(z_vars)
        a_vars_all.append(a_vars)

    # Final logits are last z_vars
    final_z = z_vars_all[-1]
    #check robustness negation: exists x in box such that argmax != pred_class
    # For pred_class, evaluate the center point with a simple numpy forward to get predicted class
    # Quick forward to get predicted class using float arithmetic:
    def float_forward(wb, x):
        a = x.copy()
        for W, b in wb:
            z = W.dot(a) + b
            a = np.maximum(0.0, z)
        return z  # final logits
    final_logits = float_forward(wb, x0)
    pred_class = int(np.argmax(final_logits))

    #add the negated property: there exists an x such that some other class >= target class
    other_idxs = [j for j in range(len(final_z)) if j != pred_class]
    s.add(Or([final_z[j] >= final_z[pred_class] for j in other_idxs]))

    t0 = time.time()
    res = s.check()
    t1 = time.time()
    elapsed = t1 - t0

    out = {"result": str(res), "time_s": elapsed, "pred_class": pred_class}
    if res == sat:
        m = s.model()
        ce = []
        for xi in x_vars:
            v = m.eval(xi)
            try:
                #ce.append(float(v.as_decimal(12)))
                ce.append(float(Fraction(v.as_fraction())))
            except:
                ce.append(float(v.as_long()))
        out["counterexample"] = ce
    return out, bounds

def detect_model_class_from_path(pth_path):
    fname = os.path.basename(pth_path).lower()
    if "small" in fname:
        return SmallNet
    elif "tiny" in fname:
        return TinyNet
    elif "wide" in fname:
        return WideNet
    else:
        raise ValueError(f"Cannot detect model type from {fname}")

def get_latest_model_pair():
    json_files = sorted(glob.glob("models/*_weights.json"))
    pth_files  = sorted(glob.glob("models/*.pth"))

    if not json_files or not pth_files:
        raise FileNotFoundError("No timestamped model files found in /models/.")

    latest_json = json_files[-1]

    # Match JSON timestamp prefix with PTH
    base = latest_json.replace("_weights.json", "")
    matching_pth = base + ".pth"

    if matching_pth in pth_files:
        return latest_json, matching_pth

    # fallback if timestamps mismatched
    print("Warning: timestamps did not match — using most recent files independently")
    return latest_json, pth_files[-1]

def load_test_points(pt_model_path, K=10):
    model_class = detect_model_class_from_path(pt_model_path)
    model = model_class()
    model.load_state_dict(torch.load(pt_model_path, map_location="cpu"))
    model.eval()

    # Select least confident points
    X_test, y_test, _ = select_least_confident_points(model, X_train, y_train, num_points=K)
    return X_test, y_test, scaler


# -------------------------------------------------------------
# Command line entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to JSON weights for SMT encoding")
    parser.add_argument("--pt-model", type=str, default=None,
                        help="Path to PyTorch .pth state_dict")
    parser.add_argument("--test-index", type=int, default=0,
                        help="Index into automatically selected test points")
    parser.add_argument("--num-test-points", type=int, default=10,
                        help="How many least-confident points to extract")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Robustness epsilon")
    parser.add_argument("--phase-mode", type=str, choices=["if", "bool"], default="if",
                        help="ReLU encoding mode")
    parser.add_argument("--timeout", type=int, default=None,
                        help="SMT solver timeout (ms)")
    args = parser.parse_args()

    # Auto-select model files if not provided
    model_json = args.model
    model_pth  = args.pt_model
    if model_json is None or model_pth is None:
        model_json, model_pth = get_latest_model_pair()
        print(f"\n\nAuto-selected model files:\n  JSON → {model_json}\n  PTH → {model_pth}")

    # Load test points
    X_test, y_test, _ = load_test_points(model_pth, K=args.num_test_points)

    if args.test_index >= len(X_test):
        raise ValueError(f"test-index {args.test_index} out of range! Select 0–{len(X_test)-1}")

    test_point = X_test[args.test_index]
    test_label = y_test[args.test_index]

    print(f"\nSelected test point #{args.test_index} "
          f"(true label = {test_label})")

    # Run verification
    print(f"\nRunning verify_model on {model_json} with eps={args.eps}, phase_mode={args.phase_mode}")
    out, bounds = verify_model(model_json, test_point, eps=args.eps,
                               phase_mode=args.phase_mode, timeout=args.timeout)
    print("\nVerification result:", out)