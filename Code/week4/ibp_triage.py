# ibp_triage.py
# Simple Interval Bound Propagation (IBP) for conv + linear layers using numpy.
# Given model JSON and a test point, compute z_lb/z_ub and a_lb/a_ub for each layer.
# These are the upper and lower activation bounds which are the smallest and largest possible values a neuron could take if the input pixels are slightly changes within a given epsilon range.

import numpy as np
import json
import argparse
from train_mnist_conv import MNISTConv
import torch

def load_json_weights(json_path):
    # Load model parameters from a JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def apply_conv_ibp(prev_lb, prev_ub, W, b, stride=1, padding=0):
    '''
    For each output pixel location (i,j) and output channel (co):
        Look at a small window in previous feature map
        Then for each input weight (w) and input pixel bounds [a,b]:
            if w>=0 -> [w*a, w*b]
            if w<0  -> [w*b, w*a]
        Then sum all contributions and add bias to give preactivation upper/lower bounds (z_ub, z_lb)
        Then apply ReLU
        Outputs bout z_* and a_* bounds for the layer
    '''
    # prev_lb, prev_ub: shape (C,H,W)
    C_out, C_in, kh, kw = W.shape
    H, Ww = prev_lb.shape[1], prev_lb.shape[2]
    # compute output shape after conv with padding and stride
    out_h = (H + 2*padding - kh)//stride + 1
    out_w = (Ww + 2*padding - kw)//stride + 1
    z_lb = np.zeros((C_out, out_h, out_w))
    z_ub = np.zeros((C_out, out_h, out_w))
    #pad prev
    if padding > 0:
        prev_lb_p = np.pad(prev_lb, ((0,0),(padding,padding),(padding,padding)), constant_values=0.0)
        prev_ub_p = np.pad(prev_ub, ((0,0),(padding,padding),(padding,padding)), constant_values=0.0)
    else:
        prev_lb_p, prev_ub_p = prev_lb, prev_ub
    for co in range(C_out):
        for i in range(out_h):
            for j in range(out_w):
                s_lb = 0.0
                s_ub = 0.0
                for ci in range(C_in):
                    for ki in range(kh):
                        for kj in range(kw):
                            w = W[co,ci,ki,kj]
                            a = prev_lb_p[ci, i+ki, j+kj]
                            b_ = prev_ub_p[ci, i+ki, j+kj]
                            if w >= 0:
                                s_lb += w * a
                                s_ub += w * b_
                            else:
                                s_lb += w * b_
                                s_ub += w * a
                z_lb[co,i,j] = s_lb + b[co]
                z_ub[co,i,j] = s_ub + b[co]
    a_lb = np.maximum(0.0, z_lb)
    a_ub = np.maximum(0.0, z_ub)
    return z_lb, z_ub, a_lb, a_ub

def apply_linear_ibp(prev_lb, prev_ub, W, b):
    '''
    For each nueron (i):
        compute linear combination of all input considering sign of weights
        Then apply ReLU
        Output z_lb, z_ub, and ReLU'd a_lb and a_ub
    '''
    # W: (out, in), prev: (in,)
    out_dim = W.shape[0]
    z_lb = np.zeros(out_dim)
    z_ub = np.zeros(out_dim)
    for i in range(out_dim):
        cl = 0.0
        cu = 0.0
        for j in range(W.shape[1]):
            w = W[i,j]
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
    return z_lb, z_ub, a_lb, a_ub

def ibp_for_json(json_path, x0, eps):
    '''
    Main analysis function:
        Set up input bounds
        Conv1 bounds
        Conv2 bounds
        Max Pool approximation - do conservatively where lower bound uses minpool for worst case and upper bound max pool for best case
        Flatten
        FC1 and FC2 bounds - the fc2 ouput logit bounds can tell if the netwrok's prediction is provably stable for that input
        Return all layers 
    '''
    data = load_json_weights(json_path)
    # x0 is (1,28,28)
    # start with input bounds
    lb = np.clip(x0 - eps, 0.0, 1.0)
    ub = np.clip(x0 + eps, 0.0, 1.0)
    #conv1
    conv1_w = np.array(data["conv1.weight"]["vals"])
    conv1_b = np.array(data["conv1.bias"]["vals"])
    z1_lb, z1_ub, a1_lb, a1_ub = apply_conv_ibp(lb, ub, conv1_w, conv1_b, stride=1, padding=1)
    #conv2
    conv2_w = np.array(data["conv2.weight"]["vals"])
    conv2_b = np.array(data["conv2.bias"]["vals"])
    z2_lb, z2_ub, a2_lb, a2_ub = apply_conv_ibp(a1_lb, a1_ub, conv2_w, conv2_b, stride=1, padding=1)
    #pool -> downsample by 2 (maxpool) over bounds: conservative -> ub = maxpool(ub), lb = minpool(lb)
    a2_lb_p = a2_lb.reshape(a2_lb.shape[0],14,2,14,2).min(axis=(2,4))
    a2_ub_p = a2_ub.reshape(a2_ub.shape[0],14,2,14,2).max(axis=(2,4))
    # flatten
    flat_lb = a2_lb_p.reshape(-1)
    flat_ub = a2_ub_p.reshape(-1)
    #fc1
    fc1_w = np.array(data["fc1.weight"]["vals"])
    fc1_b = np.array(data["fc1.bias"]["vals"])
    z3_lb, z3_ub, a3_lb, a3_ub = apply_linear_ibp(flat_lb, flat_ub, fc1_w, fc1_b)
    #fc2 (logits)
    fc2_w = np.array(data["fc2.weight"]["vals"])
    fc2_b = np.array(data["fc2.bias"]["vals"])
    #example: if class 7's lower bound is higher than all other upper bounds, its certified robust to that epsilon's perturbations
    z4_lb, z4_ub, a4_lb, a4_ub = apply_linear_ibp(a3_lb, a3_ub, fc2_w, fc2_b)
    bounds = {
        "z1_lb": z1_lb.tolist(), "z1_ub": z1_ub.tolist(),
        "z2_lb": z2_lb.tolist(), "z2_ub": z2_ub.tolist(),
        "z3_lb": z3_lb.tolist(), "z3_ub": z3_ub.tolist(),
        "z4_lb": z4_lb.tolist(), "z4_ub": z4_ub.tolist()
    }
    return bounds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, help="model json path")
    parser.add_argument("--candidates", type=str, default=None, help="candidates json")
    parser.add_argument("--eps", type=float, default=5/255.)
    args = parser.parse_args()
    if args.candidates:
        with open(args.candidates) as f:
            cands = json.load(f)
        out = []
        for c in cands:
            idx = c["index"]
            # load image from MNIST dataset same as select_candidates used
            transform = transforms = None
        print("Use ibp_triage in pipeline (called from other scripts).")
