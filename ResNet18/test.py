import torch
from ResNet18 import ResNet18
import numpy as np


output_dim = 1000
m = 10

x = np.random.random((m, 3, 32, 32))
x = torch.from_numpy(x).float()

# we half the image dim 5 times and increase the filter size to 512
model = ResNet18(x.shape[1], output_dim).float()
output = model(x)

assert output.shape[0] == m
assert output.shape[1] == output_dim
