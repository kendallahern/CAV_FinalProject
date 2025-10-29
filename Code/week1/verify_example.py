# verify_example.py

'''
Skeleton script to experiment with weights in W1 and W2.
Possibly biases too in b1 and b2,

Help to figure out where the encoding may have gone wrong and give something small to iterate on
'''

from z3 import *
import json

from train_toy_network import X_test, y_test, model


# # ---------------------------------------------------
# # Uncomment section if Testing with random weights
# # --- Example: a 2 -> 2 net (for illustration). Replace W1,b1,W2,b2 with your trained floats.
# W1 = [[0.5, -0.2], [0.3, 0.8]]   # 2x2
# b1 = [0.1, -0.05]
# W2 = [[1.2, -0.7], [-0.4, 0.9]]  # 2x2 outputs
# b2 = [0.0, 0.0]

# # original input and eps
# x0 = [0.3, -0.1]
target_class = 0  # assume model predicts class 0 at x0
# # ---------------------------------------------------

# # ---------------------------------------------------
# Uncomment section if Testing with weights from json file
with open("tiny_net_weights.json") as f:
    data = json.load(f)
W1 = data["weights"]["fc1.weight"]
b1 = data["weights"]["fc1.bias"]
W2 = data["weights"]["fc2.weight"]
b2 = data["weights"]["fc2.bias"]
W3 = data["weights"]["fc3.weight"]
b3 = data["weights"]["fc3.bias"]

# pick a test sample
x0 = X_test[0].tolist()
y0 = int(y_test[0])
print("Sample true label:", y0)
target_class = y0
# evaluate model prediction
import torch
x0_t = torch.tensor(x0, dtype=torch.float32)
pred_class = model(x0_t).argmax().item()
print("Predicted class:", pred_class)

# # ---------------------------------------------------

# original epsilon
eps = 0.05

# Z3 variables
x = [Real(f'x_{i}') for i in range(len(x0))]
# input constraints: L_inf box
s = Solver()
for i in range(len(x0)):
    s.add(x[i] >= x0[i] - eps)
    s.add(x[i] <= x0[i] + eps)

# Layer 1 affine
z1 = [Real(f'z1_{i}') for i in range(len(b1))]
a1 = [Real(f'a1_{i}') for i in range(len(b1))]
for i in range(len(b1)):
    expr = Sum([RealVal(W1[i][j]) * x[j] for j in range(len(x0))]) + RealVal(b1[i])
    s.add(z1[i] == expr)
    # ReLU
    s.add(a1[i] == If(z1[i] >= 0, z1[i], RealVal(0)))

# Layer 2 affine (outputs)
z2 = [Real(f'z2_{i}') for i in range(len(b2))]
for i in range(len(b2)):
    expr = Sum([RealVal(W2[i][j]) * a1[j] for j in range(len(b1))]) + RealVal(b2[i])
    s.add(z2[i] == expr)

# property negation: exists x in box such that some other class >= target class
other_classes = [j for j in range(len(b2)) if j != target_class]
s.add(Or([z2[j] >= z2[target_class] for j in other_classes]))

# check
res = s.check()
print("Z3 result:", res)
if res == sat:
    m = s.model()
    counter_x = [m[xi].as_decimal(10) for xi in x]
    print("Counterexample input in box:", counter_x)
    # you can convert decimals to floats, or print model directly
