from maraboupy import Marabou
import numpy as np

# small test network
network = Marabou.read_onnx("models/20251117_104023_MiniNet.onnx")
x0 = np.random.rand(1, 28, 28)  # dummy input

vals, stats = network.solve()
print(vals, stats)
