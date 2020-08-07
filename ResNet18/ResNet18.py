import torch
import torch.nn as nn
import torch.nn.functional as F

# @param input_dim - Tuple represinting the input dimension of shape (channels, height, width)
# @param output_dim - Integer representing the number of classes to predict


class ResNet18(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1_1 = nn.Conv2d(
            input_dim[0], 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(64)
        self.conv2_skip = nn.Conv2d(
            64, 128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2_skip = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_4 = nn.BatchNorm2d(128)
        self.conv3_skip = nn.Conv2d(
            128, 256, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn3_skip = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.conv4_3 = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.conv4_4 = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_4 = nn.BatchNorm2d(256)
        self.conv4_skip = nn.Conv2d(
            256, 512, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn4_skip = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5_4 = nn.BatchNorm2d(512)

        fc_dim = int(input_dim[1] * 0.5 ** 5) * 512
        self.fc = nn.Linear(fc_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        res = x

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x) + res))
        res = x
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = F.relu(self.bn2_4(self.conv2_4(x) + res))
        res = F.relu(self.bn2_skip(self.conv2_skip(x)))

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x) + res))
        res = x
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = F.relu(self.bn3_4(self.conv3_4(x) + res))
        res = F.relu(self.bn3_skip(self.conv3_skip(x)))

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x) + res))
        res = x
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = F.relu(self.bn4_4(self.conv4_4(x) + res))
        res = F.relu(self.bn4_skip(self.conv4_skip(x)))

        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x) + res))
        res = x
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = F.relu(self.bn5_4(self.conv5_4(x) + res))

        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x
