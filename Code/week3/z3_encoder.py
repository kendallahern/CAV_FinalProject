# z3_encoder.py
# Encodes MNISTConv (conv + pool + fc) into Z3 
# using Real variables and If for each neuron output (pre and post ReLU)
# Add constraints to the solver to enforce the netwroks computations
# Check whether there exists an input within a small epsilon "box" around a test image that changes the models prediction.

from z3 import Solver, Real, RealVal, If, Sum, Or
import json
import numpy as np
from fractions import Fraction
import time

#load weights from JSON filr
#saved by train_mnist_conv.py
def load_model_json(path):
    with open(path) as f:
        return json.load(f)

def rational_from_z3(val):
    # Convert z3 value to float robustly via as_fraction
    #handle boht rational/exact and decimal/approximate representations
    try:
        frac = val.as_fraction()
        return float(Fraction(frac))
    except Exception:
        s = val.as_decimal(20)
        if "?" in s:
            s = s.replace("?", "")
        return float(s)

#Encode a convolutional layer into Z3 constraints
def encode_conv_layer(solver, inputs, W, b, layer_name, padding=1, stride=1):
    ''' 
    inputs:         numpy array shape (C_in,H,W) with z3 Reals
    W:              numpy (C_out, C_in, kh, kw)
    b:              bias vector (C_out,)
    layer_name:     used to name z and a variables

    Note: the Z3 Reals in the input are the previous layer activations
    '''
    C_out, C_in, kh, kw = W.shape
    H = inputs.shape[1]
    Ww = inputs.shape[2]
    out_h = (H + 2*padding - kh)//stride + 1
    out_w = (Ww + 2*padding - kw)//stride + 1
    # pad input by zeros in arrays of Reals with python-level pads
    padded = np.empty((C_in, H + 2*padding, Ww + 2*padding), dtype=object)
    for ci in range(C_in):
        for i in range(H + 2*padding):
            for j in range(Ww + 2*padding):
                if padding <= i < H+padding and padding <= j < Ww+padding:
                    padded[ci,i,j] = inputs[ci,i-padding,j-padding]
                else:
                    padded[ci,i,j] = RealVal(0.0)
    #encode convolution operation for each output element
    z_vars = []
    a_vars = []
    for co in range(C_out):
        z_co = []
        a_co = []
        for i in range(out_h):
            for j in range(out_w):
                #linear combination (dot product) of receptive field
                terms = []
                for ci in range(C_in):
                    for ki in range(kh):
                        for kj in range(kw):
                            w = float(W[co,ci,ki,kj])
                            terms.append( RealVal(w) * padded[ci, i+ki, j+kj] )
                #create x on pre-activation variable
                z_name = f"{layer_name}_z_{co}_{i}_{j}"
                z_var = Real(z_name)
                #Constraint: z = Sum(weight*input) + bias
                solver.add(z_var == Sum(terms) + RealVal(float(b[co])))
                a_var = Real(f"{layer_name}_a_{co}_{i}_{j}")
                #create a post ReLU variable and encode ReLU constraint
                solver.add(a_var == If(z_var >= 0, z_var, RealVal(0)))
                z_co.append(z_var)
                a_co.append(a_var)
        z_vars.append(z_co)
        a_vars.append(a_co)
    # reshape outputs to numpy arrays of z3 Reals for easy access in next layer
    #return a_vars arranged as (C_out, out_h, out_w)
    a_np = np.empty((C_out, out_h, out_w), dtype=object)
    for co in range(C_out):
        k = 0
        for i in range(out_h):
            for j in range(out_w):
                a_np[co,i,j] = a_vars[co][k]
                k += 1
    return a_np

#Encodes a fully connected/dense layer into Z3
def encode_dense_layer(solver, inputs_flat, W, b, layer_name):
    '''
    inputs_flat:    list of z3 Real activations 1D
    W:              numpy array (out_dim, in_dim)
    b:              bias vector (out_dim,)
    '''
    out_dim = W.shape[0]
    z_vars = []
    for i in range(out_dim):
        terms = []
        for j in range(W.shape[1]):
            terms.append( RealVal(float(W[i,j])) * inputs_flat[j] )
        z = Real(f"{layer_name}_z_{i}")
        solver.add(z == Sum(terms) + RealVal(float(b[i])))
        a = Real(f"{layer_name}_a_{i}")
        #last fc usually logits (no ReLU), but caller chooses
        z_vars.append((z,a))
    return z_vars

#Flatten a 3D actication tensor into a 1D list for FC layer input
def flatten_conv_activation(a_np):
    # a_np: (C,H,W) of z3 Reals -> flatten list in row-major
    C,H,W = a_np.shape
    flat = []
    for c in range(C):
        for i in range(H):
            for j in range(W):
                flat.append(a_np[c,i,j])
    return flat

#verify the robustenss of a given input under epsilon perturbation
def verify_point(model_json_path, x0, eps, delta=0.0, timeout=None):
    """
    Verifies whether there exists any input (x0) in the vector space withing an epsilon radius around the input that causes the models predicted class to change by at least some delta margin

    model_json_path:    path to JSON weights
    x0:                 numpy array shape (1,28,28) float
    eps:                scalar perturbation bound
    delta:              margin (float) for classification
    timeout:            OPTIONAL
    """

    #load model and initialize Z3 solver
    m = load_model_json(model_json_path)
    solver = Solver()
    if timeout:
        solver.set("timeout", int(timeout))

    # Encode input vars with vector space constraints
    C_in = 1; H=28; Ww=28
    x_vars = np.empty((C_in,H,Ww), dtype=object)
    for i in range(H):
        for j in range(Ww):
            v = Real(f"x_0_{i}_{j}")
            solver.add(v >= RealVal(float(max(0.0, x0[0,i,j] - eps))))
            solver.add(v <= RealVal(float(min(1.0, x0[0,i,j] + eps))))
            x_vars[0,i,j] = v

    # encode conv1 layer
    W1 = np.array(m["conv1.weight"]["vals"])
    b1 = np.array(m["conv1.bias"]["vals"])
    a1 = encode_conv_layer(solver, x_vars, W1, b1, layer_name="conv1", padding=1)
    # encode conv2 layer
    W2 = np.array(m["conv2.weight"]["vals"])
    b2 = np.array(m["conv2.bias"]["vals"])
    a2 = encode_conv_layer(solver, a1, W2, b2, layer_name="conv2", padding=1)

    '''
    Max pooling layer approximation (conservative approach):
        Maxpool is linear and hard to encode exaclty so instead we "relax" it using bounds
            each var p satisfies    p >= each input in 2x2 region AND
                                    p <= sum(inputs)
        This is an over approximation, but ensures soundess AKA no false UNSATS
    '''
    C_out, H2, W2 = a2.shape
    pool_h = H2//2; pool_w = W2//2
    pooled = np.empty((C_out, pool_h, pool_w), dtype=object)
    for c in range(C_out):
        for i in range(pool_h):
            for j in range(pool_w):
                # elements: (2*i,2*j), (2*i+1,2*j), ...
                elems = [ a2[c,2*i,2*j], a2[c,2*i+1,2*j], a2[c,2*i,2*j+1], a2[c,2*i+1,2*j+1] ]
                p = Real(f"pool_{c}_{i}_{j}")
                '''
                lower bound <= p <= upper bound 
                    but we can't compute numeric bounds without IBP
                instead, constrain p >= each elem and p <= sum

                To be conservative, set p >= each elem and p <= sum(elems)
                '''
                for e in elems:
                    solver.add(p >= e)
                solver.add(p <= Sum(elems))
                pooled[c,i,j] = p

    # flatten pooled and encode fully connected layers
    flat = []
    C_p, Hp, Wp = pooled.shape
    for c in range(C_p):
        for i in range(Hp):
            for j in range(Wp):
                flat.append(pooled[c,i,j])

    # fc1 with ReLU
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
    # fc2 logits - final linear layer, no ReLU
    W_fc2 = np.array(m["fc2.weight"]["vals"])
    b_fc2 = np.array(m["fc2.bias"]["vals"])
    z_fc2 = []
    for i in range(W_fc2.shape[0]):
        terms = [ RealVal(float(W_fc2[i,j])) * z_a_fc1[j][1] for j in range(W_fc2.shape[1]) ]
        z = Real(f"fc2_z_{i}")
        solver.add(z == Sum(terms) + RealVal(float(b_fc2[i])))
        z_fc2.append(z)

    #compute predicted class on nominal (center) input
    def float_forward(mdict, x):
        '''
        A simple Numpy forward pass for the same architecture,
        want to use to get the predicted label for some input x0
        '''
        import numpy as _np
        import torch as _torch
        # reconstruct simple forward parameters using arrays
        conv1_w = _np.array(mdict["conv1.weight"]["vals"])
        conv1_b = _np.array(mdict["conv1.bias"]["vals"])
        conv2_w = _np.array(mdict["conv2.weight"]["vals"])
        conv2_b = _np.array(mdict["conv2.bias"]["vals"])
        fc1_w = _np.array(mdict["fc1.weight"]["vals"])
        fc1_b = _np.array(mdict["fc1.bias"]["vals"])
        fc2_w = _np.array(mdict["fc2.weight"]["vals"])
        fc2_b = _np.array(mdict["fc2.bias"]["vals"])
        
        #helper conv function
        def conv_forward(xn, W, b, padding=1):
            # naive conv using same padding/stride as encode_conv_layer
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

    '''
    Encode the NEGATION of the robustness property

    property:   For all x in L-inf-ball, logits[pred] >= logits[others] + delta
    negation:   Exists x s.t.            logits[pred] <= logits[others] + delta
    '''
    others = [i for i in range(len(z_fc2)) if i != pred]
    solver.add( Or([ z_fc2[i] + RealVal(float(delta)) >= z_fc2[pred] for i in others ]) )

    #solve and record results
    t0 = time.time()
    res = solver.check()
    t1 = time.time()
    out = {"result": str(res), "time_s": t1-t0}
    #if SAT, extract counterexample (adverserial input)
    if res == solver.sat:
        m = solver.model()
        #collect input counterexample
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
