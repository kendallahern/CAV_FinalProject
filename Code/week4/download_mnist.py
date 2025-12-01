import torchvision.datasets as datasets
import torchvision.transforms as T
import os

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

if __name__ == "__main__":
    data_root = "data"
    ensure_dir(data_root)

    print("Downloading MNIST to:", data_root)
    transform = T.Compose([T.ToTensor()])

    datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    print("✓ MNIST download complete.")