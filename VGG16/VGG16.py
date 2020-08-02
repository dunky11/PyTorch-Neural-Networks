import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, input_dim, output_dim):
