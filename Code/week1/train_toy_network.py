# train_toy_network.py

'''
2→8→4→2: small but nonlinear enough to make verification interesting yet solvable.

ReLU activations are piecewise linear, which SMT solvers can handle even though each ReLU introduces a linear disjunction.

The JSON export shoule be easy to load and hardcode into the Z3 script.

StandardScaler is used to normalizes data. If needed, expand to include the inverse transform for the SMT encoding so that the epsioln constrained ball is in the original feature sapce. 

'''
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

# 1. Generate small 2D dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

# 3. Define small feed-forward ReLU network
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = TinyNet()

# 4. Train it
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        preds = out.argmax(dim=1)
        acc = (preds == y_train_t).float().mean().item()
        print(f"Epoch {epoch+1}, loss={loss.item():.4f}, train_acc={acc:.3f}")

# 5. Test accuracy
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)
    acc = (preds == y_test_t).float().mean().item()
print(f"Test accuracy: {acc:.3f}")

# 6. Save model weights and biases in a JSON for Z3
weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().numpy().tolist()

# Save scale parameters (to undo normalization for Z3 if needed)
export = {
    "weights": weights,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist()
}

with open("tiny_net_weights.json", "w") as f:
    json.dump(export, f, indent=2)

print("Saved weights to tiny_net_weights.json")
