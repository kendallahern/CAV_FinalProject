# z3_encoder.py
# Encodes MNISTConv (conv + pool + fc) into Z3 using Real variables and If for ReLU.
# For each conv output element we create a Real z variable and an a variable (post-ReLU).
# This is slow for large networks, but fine for single-point verification on MNIST small convnet.

from z3 import Solver, Real, RealVal, If, Sum, Or
import json
import numpy as np
from fractions import Fraction
import time

def load_model_json(path):
    with open(path) as f:
        return json.load(f)

def rational_from_z3(val):
    # Convert z3 value to float robustly via as_fraction
    try:
        frac = val.as_fraction()
        return float(Fraction(frac))
    except Exception:
        s = val.as_decimal(20)
        if "?" in s:
            s = s.replace("?", "")
        return float(s)

def encode_conv_layer(solver, inputs, W, b, layer_name, padding=1, stride=1):
    # inputs: list of lists or numpy array shape (C_in,H,W) represented as z3 Reals (per element)
    # W: numpy (C_out, C_in, kh, kw)
    C_out, C_in, kh, kw = W.shape
    H = inputs.shape[1]
    Ww = inputs.shape[2]
    out_h = (H + 2*padding - kh)//stride + 1
    out_w = (Ww + 2*padding - kw)//stride + 1
    # pad input by zeros in arrays of Reals: we will use python-level pads
    padded = np.empty((C_in, H + 2*padding, Ww + 2*padding), dtype=object)
    for ci in range(C_in):
        for i in range(H + 2*padding):
            for j in range(Ww + 2*padding):
                if padding <= i < H+padding and padding <= j < Ww+padding:
                    padded[ci,i,j] = inputs[ci,i-padding,j-padding]
                else:
                    padded[ci,i,j] = RealVal(0.0)
    z_vars = []
    a_vars = []
    for co in range(C_out):
        z_co = []
        a_co = []
        for i in range(out_h):
            for j in range(out_w):
                terms = []
                for ci in range(C_in):
                    for ki in range(kh):
                        for kj in range(kw):
                            w = float(W[co,ci,ki,kj])
                            terms.append( RealVal(w) * padded[ci, i+ki, j+kj] )
                z_name = f"{layer_name}_z_{co}_{i}_{j}"
                z_var = Real(z_name)
                solver.add(z_var == Sum(terms) + RealVal(float(b[co])))
                a_var = Real(f"{layer_name}_a_{co}_{i}_{j}")
                # ReLU
                solver.add(a_var == If(z_var >= 0, z_var, RealVal(0)))
                z_co.append(z_var)
                a_co.append(a_var)
        z_vars.append(z_co)
        a_vars.append(a_co)
    # reshape outputs to numpy arrays of z3 Reals for next layer
    # We'll return a_vars arranged as (C_out, out_h, out_w)
    a_np = np.empty((C_out, out_h, out_w), dtype=object)
    for co in range(C_out):
        k = 0
        for i in range(out_h):
            for j in range(out_w):
                a_np[co,i,j] = a_vars[co][k]
                k += 1
    return a_np

def encode_dense_layer(solver, inputs_flat, W, b, layer_name):
    # inputs_flat: list of z3 Reals
    out_dim = W.shape[0]
    z_vars = []
    for i in range(out_dim):
        terms = []
        for j in range(W.shape[1]):
            terms.append( RealVal(float(W[i,j])) * inputs_flat[j] )
        z = Real(f"{layer_name}_z_{i}")
        solver.add(z == Sum(terms) + RealVal(float(b[i])))
        a = Real(f"{layer_name}_a_{i}")
        # last fc usually logits (no ReLU), but caller chooses
        z_vars.append((z,a))
    return z_vars

def flatten_conv_activation(a_np):
    # a_np: (C,H,W) of z3 Reals -> flatten list in row-major
    C,H,W = a_np.shape
    flat = []
    for c in range(C):
        for i in range(H):
            for j in range(W):
                flat.append(a_np[c,i,j])
    return flat

def verify_point(model_json_path, x0, eps, delta=0.0, timeout=None):
    """
    model_json_path: json file with weights saved by train_mnist_conv.py
    x0: numpy array shape (1,28,28) float
    eps: scalar
    delta: margin (float)
    """
    m = load_model_json(model_json_path)
    solver = Solver()
    if timeout:
        solver.set("timeout", int(timeout))
    # Input vars: x_{c}_{i}_{j} shape (1,28,28)
    C_in = 1; H=28; Ww=28
    x_vars = np.empty((C_in,H,Ww), dtype=object)
    for i in range(H):
        for j in range(Ww):
            v = Real(f"x_0_{i}_{j}")
            solver.add(v >= RealVal(float(max(0.0, x0[0,i,j] - eps))))
            solver.add(v <= RealVal(float(min(1.0, x0[0,i,j] + eps))))
            x_vars[0,i,j] = v

    # conv1
    W1 = np.array(m["conv1.weight"]["vals"])
    b1 = np.array(m["conv1.bias"]["vals"])
    a1 = encode_conv_layer(solver, x_vars, W1, b1, layer_name="conv1", padding=1)
    # conv2
    W2 = np.array(m["conv2.weight"]["vals"])
    b2 = np.array(m["conv2.bias"]["vals"])
    a2 = encode_conv_layer(solver, a1, W2, b2, layer_name="conv2", padding=1)
    # maxpool 2x2 -> perform conservative pooling by introducing new Reals equal to max variables? For simplicity, we'll model average pooling replacement by taking corresponding 2x2 outputs and encoding
    # For correctness we will approximate maxpool by taking a fresh variable equal to a disjunction; to keep encoding linear we will conservatively approximate by allowing the pooled value to be between min and max:
    # Create pooled variables with bounds but not exact max: this is a relaxation; might cause 'sat' more often, but will be conservative for safety proofs.
    C_out, H2, W2 = a2.shape
    pool_h = H2//2; pool_w = W2//2
    pooled = np.empty((C_out, pool_h, pool_w), dtype=object)
    for c in range(C_out):
        for i in range(pool_h):
            for j in range(pool_w):
                # elements: (2*i,2*j), (2*i+1,2*j), ...
                elems = [ a2[c,2*i,2*j], a2[c,2*i+1,2*j], a2[c,2*i,2*j+1], a2[c,2*i+1,2*j+1] ]
                p = Real(f"pool_{c}_{i}_{j}")
                # lower bound <= p <= upper bound - but we can't compute numeric bounds without IBP
                # instead, constrain p >= each elem and p <= sum? To be conservative, set p >= each elem and p <= sum(elems)
                for e in elems:
                    solver.add(p >= e)
                solver.add(p <= Sum(elems))
                pooled[c,i,j] = p

    # flatten pooled
    flat = []
    C_p, Hp, Wp = pooled.shape
    for c in range(C_p):
        for i in range(Hp):
            for j in range(Wp):
                flat.append(pooled[c,i,j])

    # fc1
    W_fc1 = np.array(m["fc1.weight"]["vals"])
    b_fc1 = np.array(m["fc1.bias"]["vals"])
    z_a_fc1 = []
    for i in range(W_fc1.shape[0]):
        terms = [ RealVal(float(W_fc1[i,j])) * flat[j] for j in range(W_fc1.shape[1]) ]
        z = Real(f"fc1_z_{i}")
        solver.add(z == Sum(terms) + RealVal(float(b_fc1[i])))
        a = Real(f"fc1_a_{i}")
        solver.add(a == If(z >= 0, z, RealVal(0)))
        z_a_fc1.append((z,a))
    # fc2 logits
    W_fc2 = np.array(m["fc2.weight"]["vals"])
    b_fc2 = np.array(m["fc2.bias"]["vals"])
    z_fc2 = []
    for i in range(W_fc2.shape[0]):
        terms = [ RealVal(float(W_fc2[i,j])) * z_a_fc1[j][1] for j in range(W_fc2.shape[1]) ]
        z = Real(f"fc2_z_{i}")
        solver.add(z == Sum(terms) + RealVal(float(b_fc2[i])))
        z_fc2.append(z)

    # get prediction on center point using float forward to select target class
    # we use numpy forward for center
    def float_forward(mdict, x):
        import numpy as _np
        import torch as _torch
        # reconstruct simple forward using arrays
        conv1_w = _np.array(mdict["conv1.weight"]["vals"])
        conv1_b = _np.array(mdict["conv1.bias"]["vals"])
        conv2_w = _np.array(mdict["conv2.weight"]["vals"])
        conv2_b = _np.array(mdict["conv2.bias"]["vals"])
        fc1_w = _np.array(mdict["fc1.weight"]["vals"])
        fc1_b = _np.array(mdict["fc1.bias"]["vals"])
        fc2_w = _np.array(mdict["fc2.weight"]["vals"])
        fc2_b = _np.array(mdict["fc2.bias"]["vals"])
        # naive conv using same padding/stride as encode_conv_layer
        def conv_forward(xn, W, b, padding=1):
            C_out, C_in, kh, kw = W.shape
            H, Ww = xn.shape[1], xn.shape[2]
            out_h = (H + 2*padding - kh)//1 + 1
            out_w = (Ww + 2*padding - kw)//1 + 1
            padded = _np.pad(xn, ((0,0),(padding,padding),(padding,padding)), mode='constant')
            out = _np.zeros((C_out, out_h, out_w))
            for co in range(C_out):
                for i in range(out_h):
                    for j in range(out_w):
                        s = 0.0
                        for ci in range(C_in):
                            for ki in range(kh):
                                for kj in range(kw):
                                    s += W[co,ci,ki,kj] * padded[ci, i+ki, j+kj]
                        out[co,i,j] = s + b[co]
            return out
        x_in = x.copy()
        x_in = x_in.reshape(1,28,28)
        z1 = conv_forward(x_in, conv1_w, conv1_b, padding=1)
        a1 = _np.maximum(0, z1)
        z2 = conv_forward(a1, conv2_w, conv2_b, padding=1)
        a2 = _np.maximum(0, z2)
        # pool 2x2 max
        a2p = a2.reshape(a2.shape[0],14,2,14,2).max(axis=(2,4))
        flat = a2p.reshape(-1)
        z3 = fc1_w.dot(flat) + fc1_b
        a3 = _np.maximum(0, z3)
        z4 = fc2_w.dot(a3) + fc2_b
        return z4
    center_logits = float_forward(load_model_json(model_json_path), x0[0])
    pred = int(np.argmax(center_logits))

    # add negated robustness: exists x within L_inf box such that some other logit >= pred + (-delta?) we require pred >= other + delta -> negation: exists other s.t. other + delta >= pred
    others = [i for i in range(len(z_fc2)) if i != pred]
    solver.add( Or([ z_fc2[i] + RealVal(float(delta)) >= z_fc2[pred] for i in others ]) )

    t0 = time.time()
    res = solver.check()
    t1 = time.time()
    out = {"result": str(res), "time_s": t1-t0}
    if res == solver.sat:
        m = solver.model()
        # collect input counterexample
        ce = []
        for i in range(28):
            row = []
            for j in range(28):
                v = m.eval(Real(f"x_0_{i}_{j}"))
                try:
                    s = v.as_fraction()
                    ce.append(float(Fraction(s)))
                except Exception:
                    s = v.as_decimal(12)
                    if "?" in s: s = s.replace("?", "")
                    ce.append(float(s))
        out["counterexample"] = ce
    return out
