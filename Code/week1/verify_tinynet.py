# verify_tinynet.py
import json
from z3 import *
import numpy as np
import torch
from train_toy_network import TinyNet, StandardScaler, X_test, y_test, scaler, X_train, y_train

from visualize_toynet import plot_decision_boundary

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
EPS = 0.05        # epsilon-ball radius (L∞)
TEST_INDEX = 0    # index of test sample to verify
GRAPH_CAPTION = ''
# --------------------------------------------------------------------

# 1. Load trained model & weights
model = TinyNet()
with open("tiny_net_weights.json", "r") as f:
    data = json.load(f)
weights = data["weights"]

# Load weights into PyTorch model for sanity check
with torch.no_grad():
    model.fc1.weight.copy_(torch.tensor(weights["fc1.weight"]))
    model.fc1.bias.copy_(torch.tensor(weights["fc1.bias"]))
    model.fc2.weight.copy_(torch.tensor(weights["fc2.weight"]))
    model.fc2.bias.copy_(torch.tensor(weights["fc2.bias"]))
    model.fc3.weight.copy_(torch.tensor(weights["fc3.weight"]))
    model.fc3.bias.copy_(torch.tensor(weights["fc3.bias"]))

model.eval()

# 2. Pick test sample and check model prediction
x0 = X_test[TEST_INDEX]
y_true = y_test[TEST_INDEX]
x0_t = torch.tensor(x0, dtype=torch.float32)
pred_class = model(x0_t).argmax().item()

print(f"Test sample index {TEST_INDEX}")
print(f"True label: {y_true}, Predicted class: {pred_class}")
print(f"Input (normalized): {x0}")

# Extract weight matrices and biases for Z3
W1, b1 = np.array(weights["fc1.weight"]), np.array(weights["fc1.bias"])
W2, b2 = np.array(weights["fc2.weight"]), np.array(weights["fc2.bias"])
W3, b3 = np.array(weights["fc3.weight"]), np.array(weights["fc3.bias"])

# 3. Create Z3 solver and variables
n_input = len(x0)
x = [Real(f'x_{i}') for i in range(n_input)]

s = Solver()

# Input box constraint (L∞)
for i in range(n_input):
    s.add(x[i] >= x0[i] - EPS)
    s.add(x[i] <= x0[i] + EPS)

# 4. Define helper functions for layers
def affine_layer(inputs, W, b, layer_name):
    z = []
    for i in range(len(b)):
        expr = Sum([RealVal(float(W[i][j])) * inputs[j] for j in range(len(inputs))]) + RealVal(float(b[i]))
        z_i = Real(f'{layer_name}_z{i}')
        s.add(z_i == expr)
        z.append(z_i)
    return z

def relu_layer(z_list, layer_name):
    a = []
    for i, z_i in enumerate(z_list):
        a_i = Real(f'{layer_name}_a{i}')
        s.add(a_i == If(z_i >= 0, z_i, RealVal(0)))
        a.append(a_i)
    return a

# 5. Encode full network
z1 = affine_layer(x, W1, b1, "L1")
a1 = relu_layer(z1, "L1")
z2 = affine_layer(a1, W2, b2, "L2")
a2 = relu_layer(z2, "L2")
z3 = affine_layer(a2, W3, b3, "L3")  # output logits

# 6. Encode negated robustness property
target = pred_class
other_classes = [j for j in range(len(z3)) if j != target]

# There exists x within ε-ball s.t. another class >= target class
s.add(Or([z3[j] >= z3[target] for j in other_classes]))

# 7. Check satisfiability
print("\nRunning Z3 solver...")
res = s.check()
print("Z3 result:", res)

if res == sat:
    m = s.model()
    counter_x = [float(m[xi].as_decimal(6)) for xi in x]
    print("\n Counterexample found!")
    print("Perturbed input:", counter_x)
    GRAPH_CAPTION = 'Counterexample found!'

    # Optionally check network’s prediction for counterexample
    x_ce = torch.tensor(counter_x, dtype=torch.float32)
    pred_ce = model(x_ce).argmax().item()
    print(f"Adversarial predicted class: {pred_ce}")
else:
    print("\n Property holds!")
    print(f"No adversarial example found within ε = {EPS}.")
    GRAPH_CAPTION = 'Property holds!'



adv = counter_x if res == sat else None
plot_decision_boundary(model, X_train, y_train, test_point=x0, eps=EPS, adv_point=adv, graph_caption=GRAPH_CAPTION)

