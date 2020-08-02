import torch
from ResNet import ResNet
import numpy as np


output_dim = 1000
m = 10

x = np.random.random((m, 3, 224, 224))
x = torch.from_numpy(x).float()

# we half the image dim 5 times and increase the filter size to 512
image_w_h = (224 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5)
fc_dim = int(image_w_h * image_w_h * 512)
model = ResNet(x.shape[1], output_dim, fc_dim).float()
output = model(x)

assert output.shape[0] == m
assert output.shape[1] == output_dim
