import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
import json, os, csv, time
from datetime import datetime
import onnx

# ----------------------------
# Utility functions
# ----------------------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

# ----------------------------
# ConvNet definition (MiniNet)
# ----------------------------
class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        # ---- Dynamically compute flattened size ----
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)   # MNIST shape
            dummy = self.pool(F.relu(self.conv1(dummy)))
            dummy = self.pool(F.relu(self.conv2(dummy)))
            self.flatten_dim = dummy.numel()
        # --------------------------------------------

        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ----------------------------
# JSON Export for SMT encoders
# ----------------------------
def export_json(model, filename):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy().tolist()
            b = module.bias.detach().cpu().numpy().tolist()
            layers.append({"type": "linear", "W": W, "b": b})
        elif isinstance(module, nn.Conv2d):
            W = module.weight.detach().cpu().numpy().tolist()
            b = module.bias.detach().cpu().numpy().tolist()
            layers.append({"type": "conv", "W": W, "b": b,
                           "stride": module.stride,
                           "padding": module.padding,
                           "kernel_size": module.kernel_size})
    with open(filename, "w") as f:
        json.dump({"layers": layers}, f, indent=2)

# ----------------------------
# Training
# ----------------------------
def train(model, loader, opt, device):
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / total

def test(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total

# ----------------------------
# Main script
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # GPU (MPS) if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Folders
    ensure_dir("models")
    ensure_dir("results/logs")

    # Data
    transform = T.Compose([T.ToTensor()])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Model
    model = MiniNet().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    print("Training MiniNet...")
    for ep in range(args.epochs):
        acc = train(model, train_loader, opt, device)
        val = test(model, test_loader, device)
        print(f"Epoch {ep+1}/{args.epochs} — Train: {acc:.4f}, Test: {val:.4f}")

    # Timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save .pth
    pth_path = f"models/{ts}_MiniNet.pth"
    torch.save(model.state_dict(), pth_path)

    # Save .json
    json_path = f"models/{ts}_MiniNet.json"
    export_json(model, json_path)

    # Save .onnx
    onnx_path = f"models/{ts}_MiniNet.onnx"
    dummy = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}
    )

    print("\nSaved:")
    print("  PTH →", pth_path)
    print("  JSON →", json_path)
    print("  ONNX →", onnx_path)

    # Log training run
    csv_path = "results/logs/train_log.csv"
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "epochs", "train_acc", "test_acc",
                        "pth", "json", "onnx"])
        w.writerow([ts, args.epochs, acc, val, pth_path, json_path, onnx_path])

    print("\n✓ Training complete and logged.")