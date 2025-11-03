# train_mnist_conv.py

#this python file trains a small MNIST ConvNet and exports .pth and a JSON of weights for Z3 encoding in next step
import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import csv

# ----------------------
# Model - CNN for the MNIST dataset
#       - 28x28 grayscale images of handwritten digits 0-9 so 10 classifications
# ----------------------
class MNISTConv(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1x28x28
        # think (Conv2d, in_channels, out_channels...)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 8x28x28
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) #16x28x28
        self.pool  = nn.MaxPool2d(2)  # halves spatial dims but keeps most important features
        #fully connected/dense layers gets us from 16*14*16=3136 feautures per image to 64 to 10
        self.fc1   = nn.Linear(16*14*14, 64)
        self.fc2   = nn.Linear(64, 10)
        #introduce non-linearity, allow model to learn complex patterns
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))  # -> 16 x 14 x 14
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------
# Other Functions
# ----------------------
def export_weights_to_json(model, json_path):
    """
    Save all parameters with shapes to JSON.
    Keep tensors as nested lists; conv weights retain (out, in, kh, kw)
    """
    obj = {}
    for name, param in model.state_dict().items():
        obj[name] = {
            "shape": list(param.shape),
            "vals": param.detach().cpu().numpy().tolist()
        }
    with open(json_path, "w") as f:
        json.dump(obj, f)
    print("Exported weights to", json_path)

# ----------------------
# Train the model
# ----------------------
def train_and_save(out_prefix, epochs=5):
    #set device for computation and load the dataset
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Device:", device)
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    testset  = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    #initialize the model w/ Adam optimizer and Cross Entropy Loss which is suitable for multi-class classification
    model = MNISTConv().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    #training loop
    for ep in range(epochs):
        #set model to train
        model.train()
        #accumulate total loss for average calculation
        running = 0.0

        for xb, yb in trainloader:                      #iterate over mini-batches
            xb, yb = xb.to(device), yb.to(device)       # move data to device
            opt.zero_grad()                             # reset gradients
            out = model(xb)                             #forward pass (predictions)
            loss = crit(out, yb)                        #compute loss
            loss.backward()                             #backpropagation
            opt.step()                                  # update weights
            running += loss.item() * xb.size(0)         #accumulate batch loss (scaled by batch size)

        print(f"Epoch {ep+1}/{epochs} avg loss {running/len(trainset):.4f}")

    # set model to eval mode
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():                           # disable gradient tracking
        for xb, yb in testloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)               #get class with the highest score
            correct += (preds == yb).sum().item()   #count number of correct predictions
            total += yb.size(0)                     #total number of samples
    test_acc = correct / total
    print("Test acc:", test_acc)

    # save
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pth = f"models/{timestamp}_mnist_conv.pth"
    jsonf = f"models/{timestamp}_mnist_conv.json"
    onnxf = f"models/{timestamp}_mnist_conv.onnx"
    torch.save(model.state_dict(), pth)
    export_weights_to_json(model, jsonf)
    
    # export onnx (optional)
    dummy = torch.randn(1,1,28,28).to(device)
    try:
        torch.onnx.export(model, dummy, onnxf, opset_version=12)
        print("Exported ONNX to", onnxf)
    except Exception as e:
        print("ONNX export failed:", e)

    # log training
    os.makedirs("results", exist_ok=True)
    logfile = "results/training_log.csv"
    write_header = not os.path.exists(logfile)
    with open(logfile, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp","model","test_acc","pth","json","onnx"])
        writer.writerow([timestamp,"mnist_conv",test_acc,pth,jsonf,onnxf])

    return pth, jsonf

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    pth, jsonf = train_and_save(out_prefix="models/mnist_conv", epochs=5)
    print("Saved:", pth, jsonf)
