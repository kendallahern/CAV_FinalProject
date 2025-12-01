from z3 import *
import numpy as np
import json
from fractions import Fraction
import time

# --------------------------
# Utility
# --------------------------
def rational_from_z3(val):
    try:
        frac = val.as_fraction()
        return float(Fraction(frac))
    except Exception:
        s = val.as_decimal(20)
        if "?" in s:
            s = s.replace("?", "")
        return float(s)

# --------------------------
# Helper: encode conv layer for Z3
# --------------------------
def encode_conv_layer(solver, inputs, W, b, layer_name, padding=2, stride=1):
    """
    Encode a convolutional layer with ReLU for Z3.
    inputs: np.array of shape (C_in, H, W) containing Z3 Reals
    W: weight array (C_out, C_in, kh, kw)
    b: bias array (C_out,)
    Returns: np.array of shape (C_out, H_out, W_out) with Z3 Reals after ReLU
    """
    C_out, C_in, kh, kw = W.shape
    H, Ww = inputs.shape[1], inputs.shape[2]
    H_out = (H + 2*padding - kh)//stride + 1
    W_out = (Ww + 2*padding - kw)//stride + 1

    # Pad input
    padded = np.empty((C_in, H + 2*padding, Ww + 2*padding), dtype=object)
    for ci in range(C_in):
        for i in range(H + 2*padding):
            for j in range(Ww + 2*padding):
                if padding <= i < H + padding and padding <= j < Ww + padding:
                    padded[ci,i,j] = inputs[ci,i-padding,j-padding]
                else:
                    padded[ci,i,j] = RealVal(0.0)

    # Convolution + ReLU
    out = np.empty((C_out, H_out, W_out), dtype=object)
    for co in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                terms = []
                for ci in range(C_in):
                    for ki in range(kh):
                        for kj in range(kw):
                            terms.append( RealVal(float(W[co,ci,ki,kj])) * padded[ci,i+ki,j+kj] )
                z = Real(f"{layer_name}_z_{co}_{i}_{j}")
                solver.add(z == Sum(terms) + RealVal(float(b[co])))
                a = Real(f"{layer_name}_a_{co}_{i}_{j}")
                solver.add(a == If(z >= 0, z, RealVal(0)))
                out[co,i,j] = a
    return out

# --------------------------
# Helper: 2x2 maxpool for Z3
# --------------------------
def encode_maxpool2x2(solver, x, layer_name):
    C, H, W = x.shape
    H_out, W_out = H//2, W//2
    out = np.empty((C, H_out, W_out), dtype=object)
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                elems = [x[c,2*i,2*j], x[c,2*i+1,2*j], x[c,2*i,2*j+1], x[c,2*i+1,2*j+1]]
                p = Real(f"{layer_name}_pool_{c}_{i}_{j}")
                for e in elems:
                    solver.add(p >= e)
                solver.add(p <= Sum(elems))
                out[c,i,j] = p
    return out

# # --------------------------
# # Forward pass in NumPy for logits
# # --------------------------
# def forward_numpy(x, m):
#     """
#     Numpy forward pass for MiniNet using JSON model m.
#     x: (28,28) or (1,28,28)
#     m: model dict from JSON
#     """
#     if x.ndim == 2:
#         x = x.reshape(1,28,28)
    
#     # --- conv1 ---
#     W1 = np.array(m["layers"][0]["W"])
#     b1 = np.array(m["layers"][0]["b"])
#     C_out1, C_in1, kh1, kw1 = W1.shape
#     padded1 = np.pad(x, ((0,0),(2,2),(2,2)), mode='constant')
#     a1 = np.zeros((C_out1,28,28))
#     for co in range(C_out1):
#         for i in range(28):
#             for j in range(28):
#                 s = sum(W1[co,ci,ki, kj]*padded1[ci,i+ki,j+kj] 
#                         for ci in range(C_in1) for ki in range(kh1) for kj in range(kw1))
#                 a1[co,i,j] = max(0, s + b1[co])

#     # --- maxpool1 2x2 ---
#     a1p = np.zeros((C_out1,14,14))
#     for c in range(C_out1):
#         for i in range(14):
#             for j in range(14):
#                 a1p[c,i,j] = max(a1[c,2*i,2*j], a1[c,2*i+1,2*j],
#                                   a1[c,2*i,2*j+1], a1[c,2*i+1,2*j+1])

#     # --- conv2 ---
#     W2 = np.array(m["layers"][1]["W"])
#     b2 = np.array(m["layers"][1]["b"])
#     C_out2, C_in2, kh2, kw2 = W2.shape
#     padded2 = np.pad(a1p, ((0,0),(2,2),(2,2)), mode='constant')
#     a2 = np.zeros((C_out2,14,14))
#     for co in range(C_out2):
#         for i in range(14):
#             for j in range(14):
#                 s = sum(W2[co,ci,ki,kj]*padded2[ci,i+ki,j+kj] 
#                         for ci in range(C_in2) for ki in range(kh2) for kj in range(kw2))
#                 a2[co,i,j] = max(0, s + b2[co])

#     # --- maxpool2 2x2 ---
#     a2p = np.zeros((C_out2,7,7))
#     for c in range(C_out2):
#         for i in range(7):
#             for j in range(7):
#                 a2p[c,i,j] = max(a2[c,2*i,2*j], a2[c,2*i+1,2*j],
#                                   a2[c,2*i,2*j+1], a2[c,2*i+1,2*j+1])

#     # --- flatten ---
#     flatf = a2p.flatten()  # 32*7*7 = 1568

#     # --- fc1 ---
#     W_fc1 = np.array(m["layers"][2]["W"])
#     b_fc1 = np.array(m["layers"][2]["b"])
#     z3 = W_fc1 @ flatf + b_fc1
#     a3 = np.maximum(0, z3)

#     # --- fc2 ---
#     W_fc2 = np.array(m["layers"][3]["W"])
#     b_fc2 = np.array(m["layers"][3]["b"])
#     z4 = W_fc2 @ a3 + b_fc2
#     return z4

# --------------------------
# Forward pass in NumPy for logits (safe version)
# --------------------------
def forward_numpy(x, m):
    """
    Numpy forward pass for MiniNet using JSON model m.
    Safe version: uses float32, clips activations to avoid overflow.
    x: (28,28) or (1,28,28)
    m: model dict from JSON
    """
    import numpy as np

    # Ensure input shape (1,28,28)
    if x.ndim == 2:
        x = x.reshape(1, 28, 28)
    x = x.astype(np.float32)

    # --- conv1 ---
    W1 = np.array(m["layers"][0]["W"], dtype=np.float32)
    b1 = np.array(m["layers"][0]["b"], dtype=np.float32)
    C_out1, C_in1, kh1, kw1 = W1.shape
    H, Ww = x.shape[1], x.shape[2]
    padded1 = np.pad(x, ((0,0),(2,2),(2,2)), mode='constant').astype(np.float32)
    a1 = np.zeros((C_out1, H, Ww), dtype=np.float32)
    for co in range(C_out1):
        for i in range(H):
            for j in range(Ww):
                s = 0.0
                for ci in range(C_in1):
                    for ki in range(kh1):
                        for kj in range(kw1):
                            s += W1[co, ci, ki, kj] * padded1[ci, i+ki, j+kj]
                a1[co, i, j] = np.clip(max(0.0, s + b1[co]), 0, 1e3)

    # --- conv2 ---
    W2 = np.array(m["layers"][1]["W"], dtype=np.float32)
    b2 = np.array(m["layers"][1]["b"], dtype=np.float32)
    C_out2, C_in2, kh2, kw2 = W2.shape
    padded2 = np.pad(a1, ((0,0),(2,2),(2,2)), mode='constant').astype(np.float32)
    a2 = np.zeros((C_out2, H, Ww), dtype=np.float32)
    for co in range(C_out2):
        for i in range(H):
            for j in range(Ww):
                s = 0.0
                for ci in range(C_in2):
                    for ki in range(kh2):
                        for kj in range(kw2):
                            s += W2[co, ci, ki, kj] * padded2[ci, i+ki, j+kj]
                a2[co, i, j] = np.clip(max(0.0, s + b2[co]), 0, 1e3)

    # --- maxpool 2x2 ---
    H_pool, W_pool = H // 2, Ww // 2
    a2p = np.zeros((C_out2, H_pool, W_pool), dtype=np.float32)
    for c in range(C_out2):
        for i in range(H_pool):
            for j in range(W_pool):
                a2p[c,i,j] = np.clip(
                    max(
                        a2[c,2*i,2*j], a2[c,2*i+1,2*j],
                        a2[c,2*i,2*j+1], a2[c,2*i+1,2*j+1]
                    ),
                    0, 1e3
                )

    # -


# --------------------------
# MiniNet SMT Verification
# --------------------------
def mini_verify_point(model_json_path, x0, eps, delta=0.0, timeout=None):
    """
    Verify a MiniNet model (from exported JSON) at input x0 with L-inf perturbation eps.
    """
    # Load model JSON
    with open(model_json_path) as f:
        model_json = json.load(f)

    solver = Solver()
    if timeout:
        solver.set("timeout", int(timeout))

    # --- Encode input variables ---
    C_in, H, W = 1,28,28
    x_vars = np.empty((C_in,H,W), dtype=object)
    for i in range(H):
        for j in range(W):
            v = Real(f"x_0_{i}_{j}")
            solver.add(v >= RealVal(float(max(0.0, x0[0,i,j]-eps))))
            solver.add(v <= RealVal(float(min(1.0, x0[0,i,j]+eps))))
            x_vars[0,i,j] = v

    # --- Encode conv layers + pooling ---
    a1 = encode_conv_layer(solver, x_vars, np.array(model_json["layers"][0]["W"]),
                           np.array(model_json["layers"][0]["b"]), "conv1")
    a1p = encode_maxpool2x2(solver, a1, "pool1")
    a2 = encode_conv_layer(solver, a1p, np.array(model_json["layers"][1]["W"]),
                           np.array(model_json["layers"][1]["b"]), "conv2")
    a2p = encode_maxpool2x2(solver, a2, "pool2")

    # --- Flatten ---
    flat = a2p.flatten()

    # --- FC1 with ReLU ---
    W_fc1 = np.array(model_json["layers"][2]["W"])
    b_fc1 = np.array(model_json["layers"][2]["b"])
    z_a_fc1 = []
    for i in range(W_fc1.shape[0]):
        terms = [RealVal(float(W_fc1[i,j])) * flat[j] for j in range(W_fc1.shape[1])]
        z = Real(f"fc1_z_{i}")
        solver.add(z == Sum(terms) + RealVal(float(b_fc1[i])))
        a = Real(f"fc1_a_{i}")
        solver.add(a == If(z >= 0, z, RealVal(0)))
        z_a_fc1.append((z,a))

    # --- FC2 logits (linear, no ReLU) ---
    W_fc2 = np.array(model_json["layers"][3]["W"])
    b_fc2 = np.array(model_json["layers"][3]["b"])
    z_fc2 = []
    for i in range(W_fc2.shape[0]):
        terms = [ RealVal(float(W_fc2[i,j])) * z_a_fc1[j][1] for j in range(W_fc2.shape[1]) ]
        z = Real(f"fc2_z_{i}")
        solver.add(z == Sum(terms) + RealVal(float(b_fc2[i])))
        z_fc2.append(z)

    # --- Nominal prediction ---
    center_logits = forward_numpy(x0[0], model_json)
    pred = int(np.argmax(center_logits))

    # --- Encode negation of robustness property ---
    others = [i for i in range(len(z_fc2)) if i != pred]
    solver.add( Or([ z_fc2[i] + RealVal(float(delta)) >= z_fc2[pred] for i in others ]) )

    # --- Solve ---
    t0 = time.time()
    res = solver.check()
    t1 = time.time()

    out = {"result": str(res), "time_s": t1-t0}
    if res == sat:
        m = solver.model()
        ce = []
        for i in range(28):
            for j in range(28):
                v = m.eval(Real(f"x_0_{i}_{j}"))
                ce.append(rational_from_z3(v))
        out["counterexample"] = ce

    return out
