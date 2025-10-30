# smt_encoder.py
# Modular SMT encoder using Z3. Exposes verify_local_robustness(...) which returns
# a small dict: result, time_s, counterexample (if any), num_ambiguous.

from z3 import *
import numpy as np
import time
from fractions import Fraction

def ibp_bounds(weights_biases, x0, eps):
    """
    Interval bound propagation (IBP): conservative bounds for each pre-activation z and post-activation a.
    weights_biases: list of (W, b) where W: (out_dim, in_dim)
    x0: numpy array of input (normalized)
    eps: scalar (L_inf)
    Returns list of dicts: each contains z_lb, z_ub, a_lb, a_ub (numpy arrays)
    """
    lb = x0 - eps
    ub = x0 + eps
    bounds = []
    prev_lb, prev_ub = lb.copy(), ub.copy()
    for (W, b) in weights_biases:
        W = np.array(W)
        b = np.array(b)
        out_dim = W.shape[0]
        z_lb = np.zeros(out_dim, dtype=float)
        z_ub = np.zeros(out_dim, dtype=float)
        for i in range(out_dim):
            # sum contributions
            cl = 0.0
            cu = 0.0
            for j in range(W.shape[1]):
                w = W[i, j]
                if w >= 0:
                    cl += w * prev_lb[j]
                    cu += w * prev_ub[j]
                else:
                    cl += w * prev_ub[j]
                    cu += w * prev_lb[j]
            z_lb[i] = cl + b[i]
            z_ub[i] = cu + b[i]
        a_lb = np.maximum(0.0, z_lb)
        a_ub = np.maximum(0.0, z_ub)
        bounds.append({"z_lb": z_lb, "z_ub": z_ub, "a_lb": a_lb, "a_ub": a_ub})
        prev_lb, prev_ub = a_lb, a_ub
    return bounds

def add_affine_constraints(solver, inputs, W, b, layer_name):
    """
    inputs: list of Z3 Reals
    W: numpy array (out_dim, in_dim)
    b: numpy array (out_dim,)
    returns list of z vars
    """
    z_vars = []
    for i in range(W.shape[0]):
        expr = Sum([RealVal(float(W[i, j])) * inputs[j] for j in range(W.shape[1])]) + RealVal(float(b[i]))
        z_i = Real(f"{layer_name}_z{i}")
        solver.add(z_i == expr)
        z_vars.append(z_i)
    return z_vars

def encode_network_with_ibp(solver, weights_biases, x_vars, bounds, phase_mode="if"):
    """
    Encodes the full network into the solver using IBP results to simplify ReLU encodings.
    Returns final_z_vars (list) and ambiguous_count.
    phase_mode: "if" or "bool"
    """
    current_inputs = x_vars
    ambiguous_count = 0
    z_layers = []
    a_layers = []

    for layer_idx, ((W, b), layer_bounds) in enumerate(zip(weights_biases, bounds)):
        layer_name = f"L{layer_idx+1}"
        z_vars = add_affine_constraints(solver, current_inputs, W, b, layer_name)
        a_vars = [Real(f"{layer_name}_a{i}") for i in range(len(z_vars))]
        z_lb = layer_bounds["z_lb"]
        z_ub = layer_bounds["z_ub"]
        for i, z_i in enumerate(z_vars):
            if z_lb[i] >= 0.0:
                # always active: a == z
                solver.add(a_vars[i] == z_i)
            elif z_ub[i] <= 0.0:
                # always inactive: a == 0
                solver.add(a_vars[i] == RealVal(0))
                # tighten: enforce z <= z_ub
                solver.add(z_i <= RealVal(float(z_ub[i])))
            else:
                # ambiguous
                ambiguous_count += 1
                if phase_mode == "if":
                    solver.add(a_vars[i] == If(z_i >= 0, z_i, RealVal(0)))
                else:
                    phase = Bool(f"{layer_name}_phase{i}")
                    solver.add(Implies(phase, a_vars[i] == z_i))
                    solver.add(Implies(phase, z_i >= 0))
                    solver.add(Implies(Not(phase), a_vars[i] == RealVal(0)))
                    solver.add(Implies(Not(phase), z_i <= 0))
        current_inputs = a_vars
        z_layers.append(z_vars)
        a_layers.append(a_vars)
    return z_layers[-1], ambiguous_count

def verify_local_robustness(weights_biases, x0, eps, delta=0.0, phase_mode="if", timeout_ms=None):
    """
    weights_biases: list of (W,b) where both W,b are numpy arrays (floats)
    x0: numpy array (normalized input)
    eps: scalar (L_inf)
    delta: decision margin (float) - require target >= other + delta
    phase_mode: "if" or "bool"
    timeout_ms: if provided, set solver timeout in milliseconds
    Returns: dict with keys: result ('sat'/'unsat'/'unknown'), time_s (float), counterexample (list) or None, ambiguous_count
    """
    t0 = time.time()
    # prepare solver
    s = Solver()
    if timeout_ms:
        s.set("timeout", int(timeout_ms))

    # IBP
    bounds = ibp_bounds(weights_biases, x0, eps)

    # input vars
    n_in = x0.shape[0]
    x_vars = [Real(f"x_{i}") for i in range(n_in)]
    for i in range(n_in):
        s.add(x_vars[i] >= RealVal(float(x0[i] - eps)))
        s.add(x_vars[i] <= RealVal(float(x0[i] + eps)))

    # encode network
    final_z_vars, ambiguous = encode_network_with_ibp(s, weights_biases, x_vars, bounds, phase_mode=phase_mode)

    # get predicted class at center (float forward)
    def float_forward(wb, x):
        a = x.copy()
        last_z = None
        for (W, b) in wb:
            z = W.dot(a) + b
            last_z = z
            a = np.maximum(0.0, z)
        return last_z
    logits = float_forward(weights_biases, x0)
    pred = int(np.argmax(logits))
    others = [j for j in range(len(final_z_vars)) if j != pred]

    # add margin negation: exists x such that some other logit >= pred_logit - delta? WAIT: we want other >= pred - delta -> equivalently pred <= other + delta
    # To find counterexample where pred is not >= other + delta, add Or( final_z[j] + delta >= final_z[pred] )
    # But clearer: require other >= pred - delta => other + delta >= pred
    s.add(Or([final_z_vars[j] + RealVal(float(delta)) >= final_z_vars[pred] for j in others]))

    t_before = time.time()
    res = s.check()
    t_after = time.time()
    elapsed = (t_after - t_before)
    out = {"result": str(res), "time_s": elapsed, "pred_class": pred, "ambiguous": ambiguous}
    if res == sat:
        m = s.model()
        ce = []
        for xi in x_vars:
            v = m.eval(xi)
            # try decimal conversion; fallback to as_long
            try:
                ce.append(float(v.as_decimal(12)))
            except Exception:
                try:
                    ce.append(float(Fraction(v.as_fraction())))
                    #ce.append(float(v.as_long()))
                except Exception:
                    # final fallback: string parse
                    ce.append(float(str(v)))
        out["counterexample"] = ce
    t1 = time.time()
    out["total_time_s"] = t1 - t0
    return out
