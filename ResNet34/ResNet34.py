import torch
import torch.nn as nn
import torch.nn.functional as F


# @param input_channels - Number of channels in the input image,
# input image must be in shape (channels, width, height)
# @param num_classes - Number of classes to predict
class ResNet34(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(
            input_channels, 64, kernel_size=(7, 7), padding=3, stride=2, bias=False)
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm2_4 = nn.BatchNorm2d(64)
        self.conv2_5 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm2_5 = nn.BatchNorm2d(64)
        self.conv2_6 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm2_6 = nn.BatchNorm2d(64)
        self.conv2_downsample = nn.Conv2d(
            64, 128, kernel_size=(1, 1), stride=2, bias=False)
        self.conv2_downsample_batchnorm = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(
            64, 128, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.batchnorm3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm3_4 = nn.BatchNorm2d(128)
        self.conv3_5 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm3_5 = nn.BatchNorm2d(128)
        self.conv3_6 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm3_6 = nn.BatchNorm2d(128)
        self.conv3_7 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm3_7 = nn.BatchNorm2d(128)
        self.conv3_8 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm3_8 = nn.BatchNorm2d(128)
        self.conv3_downsample = nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=2, bias=False)
        self.conv3_downsample_batchnorm = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(
            128, 256, kernel_size=(3, 3), padding=1, stride=2, bias=False)
        self.batchnorm4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_2 = nn.BatchNorm2d(256)
        self.conv4_3 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_3 = nn.BatchNorm2d(256)
        self.conv4_4 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_4 = nn.BatchNorm2d(256)
        self.conv4_5 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_5 = nn.BatchNorm2d(256)
        self.conv4_6 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_6 = nn.BatchNorm2d(256)
        self.conv4_7 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_7 = nn.BatchNorm2d(256)
        self.conv4_8 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_8 = nn.BatchNorm2d(256)
        self.conv4_9 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_9 = nn.BatchNorm2d(256)
        self.conv4_10 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_10 = nn.BatchNorm2d(256)
        self.conv4_11 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_11 = nn.BatchNorm2d(256)
        self.conv4_12 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4_12 = nn.BatchNorm2d(256)
        self.conv4_downsample = nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=2, bias=False)
        self.conv4_downsample_batchnorm = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(
            256, 512, kernel_size=(3, 3), padding=1, stride=2, bias=False)
        self.batchnorm5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm5_4 = nn.BatchNorm2d(512)
        self.conv5_5 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm5_5 = nn.BatchNorm2d(512)
        self.conv5_6 = nn.Conv2d(
            512, 512, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm5_6 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.batchnorm1_1(self.conv1_1(x)))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)
        res = x

        x = self.relu(self.batchnorm2_1(self.conv2_1(x)))
        x = self.relu(self.batchnorm2_2(self.conv2_2(x) + res))
        res = x
        x = self.relu(self.batchnorm2_3(self.conv2_3(x)))
        x = self.relu(self.batchnorm2_4(self.conv2_4(x) + res))
        res = x
        x = self.relu(self.batchnorm2_5(self.conv2_5(x)))
        x = self.relu(self.batchnorm2_6(self.conv2_6(x) + res))
        res = self.relu(self.conv2_downsample_batchnorm(
            self.conv2_downsample(x)))

        x = self.relu(self.batchnorm3_1(self.conv3_1(x)))
        x = self.relu(self.batchnorm3_2(self.conv3_2(x) + res))
        res = x
        x = self.relu(self.batchnorm3_3(self.conv3_3(x)))
        x = self.relu(self.batchnorm3_4(self.conv3_4(x) + res))
        res = x
        x = self.relu(self.batchnorm3_5(self.conv3_5(x)))
        x = self.relu(self.batchnorm3_6(self.conv3_6(x) + res))
        res = x
        x = self.relu(self.batchnorm3_7(self.conv3_7(x)))
        x = self.relu(self.batchnorm3_8(self.conv3_8(x) + res))
        res = self.relu(self.conv3_downsample_batchnorm(
            self.conv3_downsample(x)))

        x = self.relu(self.batchnorm4_1(self.conv4_1(x)))
        x = self.relu(self.batchnorm4_2(self.conv4_2(x) + res))
        res = x
        x = self.relu(self.batchnorm4_3(self.conv4_3(x)))
        x = self.relu(self.batchnorm4_4(self.conv4_4(x) + res))
        res = x
        x = self.relu(self.batchnorm4_5(self.conv4_5(x)))
        x = self.relu(self.batchnorm4_6(self.conv4_6(x) + res))
        res = x
        x = self.relu(self.batchnorm4_7(self.conv4_7(x)))
        x = self.relu(self.batchnorm4_8(self.conv4_8(x) + res))
        res = x
        x = self.relu(self.batchnorm4_9(self.conv4_9(x)))
        x = self.relu(self.batchnorm4_10(self.conv4_10(x) + res))
        res = x
        x = self.relu(self.batchnorm4_11(self.conv4_11(x)))
        x = self.relu(self.batchnorm4_12(self.conv4_12(x) + res))
        res = self.relu(self.conv4_downsample_batchnorm(
            self.conv4_downsample(x)))

        x = self.relu(self.batchnorm5_1(self.conv5_1(x)))
        x = self.relu(self.batchnorm5_2(self.conv5_2(x) + res))
        res = x
        x = self.relu(self.batchnorm5_3(self.conv5_3(x) + res))
        x = self.relu(self.batchnorm5_4(self.conv5_4(x) + res))
        res = x
        x = self.relu(self.batchnorm5_5(self.conv5_5(x) + res))
        x = self.relu(self.batchnorm5_6(self.conv5_6(x) + res))

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 512)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x
