# train_variants.py
"""
Investigae how model size affects the verifiability, at least as a small scale

Train 3 small networks:
 - small: 2 -> 4 -> 2
 - tiny : 2 -> 8 -> 4 -> 2 (what was done in week 1)
 - wide: 2 -> 16 -> 8 -> 2

Saves weights into JSON files:
 - tiny_net_weights.json (from week1)
 - small_net_weights.json
 - wide_net_weights.json

Also saves a metadata file models_summary.json with test accuracies.
"""
import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np, json, os
import os
import csv
from datetime import datetime

# use the make moons dataset again to start
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

#classes for each of the different models
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

class WideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#use cross entropy loss and Adam optimizer for all
#learning rate and num epochs are default right now but may need to update
def train_model(model, epochs=200, lr=0.01):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(X_train_t)
        loss = crit(out, y_train_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train_t).argmax(dim=1) == y_train_t).float().mean().item()
        test_acc  = (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
    return model, train_acc, test_acc

def save_weights(model, filename, scaler):
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy().tolist()
    export = {"weights": weights,
              "scaler_mean": scaler.mean_.tolist(),
              "scaler_scale": scaler.scale_.tolist()}
    with open(filename, "w") as f:
        json.dump(export, f, indent=2)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

logfile = "results/training_log.csv"
write_header = not os.path.exists(logfile)

def log_results(row):
    with open(logfile, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "model_name", "train_acc", "test_acc", "json_file", "pth_file"])
        writer.writerow(row)

def train_and_save(model, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, tr_acc, te_acc = train_model(model)

    # File names with timestamp
    json_file = f"models/{timestamp}_{name}_weights.json"
    pth_file = f"models/{timestamp}_{name}.pth"

    # Save SMT-exportable weights
    save_weights(model, json_file, scaler)

    # Save PyTorch weights
    torch.save(model.state_dict(), pth_file)

    # Log entry
    log_results([timestamp, name, tr_acc, te_acc, json_file, pth_file])

    print(f"[SAVED] {name}: JSON → {json_file}, PTH → {pth_file}")
    return tr_acc, te_acc, json_file, pth_file

# Train all variants
summary = {}

summary["small"]  = train_and_save(SmallNet(), "small_net")
summary["tiny"] = train_and_save(TinyNet(), "tiny_net")
summary["wide"]  = train_and_save(WideNet(), "wide_net")

print("\nTraining summary:", summary)

#summary = {}

# # small
# m_small = SmallNet()
# m_small, tr, te = train_model(m_small)
# save_weights(m_small, "models/small_net_weights.json", scaler)
# summary["small"] = {"train_acc": tr, "test_acc": te, "file": "models/small_net_weights.json"}

# # tiny (Week1)
# m_tiny = TinyNet()
# m_tiny, tr, te = train_model(m_tiny)
# save_weights(m_tiny, "models/tiny_net_weights.json", scaler)
# summary["tiny"] = {"train_acc": tr, "test_acc": te, "file": "models/tiny_net_weights.json"}

# # wide
# m_wide = WideNet()
# m_wide, tr, te = train_model(m_wide)
# save_weights(m_wide, "models/wide_net_weights.json", scaler)
# summary["wide"] = {"train_acc": tr, "test_acc": te, "file": "models/wide_net_weights.json"}

# with open("models/models_summary.json", "w") as f:
#     json.dump(summary, f, indent=2)

# print("Trained models and saved weights under models/*.json")
