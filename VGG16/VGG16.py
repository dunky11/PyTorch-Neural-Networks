import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, input_shape: tuple, output_dim: int):
        super().__init__()
        fc_dim = int((input_shape[1] * 0.5**5) *
                     (input_shape[2] * 0.5 ** 5) * 512)
        self.maxpool = nn.MaxPool2d((2, 2), 2)
        self.relu = nn.ReLU()
        self.conv_1_1 = nn.Conv2d(
            input_shape[0], 64, kernel_size=(3, 3), padding=1)
        self.conv_1_2 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1)
        self.conv_2_1 = nn.Conv2d(
            64, 128, kernel_size=(3, 3), padding=1)
        self.conv_2_2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1)
        self.conv_3_1 = nn.Conv2d(
            128, 256, kernel_size=(3, 3), padding=1)
        self.conv_3_2 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1)
        self.conv_3_3 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1)
        self.conv_4_1 = nn.Conv2d(
            256, 512, kernel_size=(3, 3), padding=1)
        self.conv_4_2 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1)
        self.conv_4_3 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1)
        self.conv_5_1 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1)
        self.conv_5_2 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1)
        self.conv_5_3 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(
            fc_dim, 4096)
        self.fc2 = nn.Linear(
            4096, 4096)
        self.fc3 = nn.Linear(
            4096, output_dim)

    def forward(self, x):
        x = self.relu(self.conv_1_1(x))
        x = self.relu(self.conv_1_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_2_1(x))
        x = self.relu(self.conv_2_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_3_1(x))
        x = self.relu(self.conv_3_2(x))
        x = self.relu(self.conv_3_3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_4_1(x))
        x = self.relu(self.conv_4_2(x))
        x = self.relu(self.conv_4_3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_5_1(x))
        x = self.relu(self.conv_5_2(x))
        x = self.relu(self.conv_5_3(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = nn.functional.softmax(x, dim=1)
        return x
