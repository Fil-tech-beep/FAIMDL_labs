import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        """
        AlexNet-style CNN + BatchNorm

        Input: (3, 227, 227)

        conv1:  (3, 227, 227)   -> (64, 56, 56)
        pool1:  (64, 56, 56)    -> (64, 27, 27)

        conv2:  (64, 27, 27)    -> (192, 27, 27)
        pool2:  (192, 27, 27)   -> (192, 13, 13)

        conv3:  (192, 13, 13)   -> (384, 13, 13)
        conv4:  (384, 13, 13)   -> (256, 13, 13)
        conv5:  (256, 13, 13)   -> (256, 13, 13)
        pool3:  (256, 13, 13)   -> (256, 6, 6)

        flatten: 256 * 6 * 6 = 9216

        fc1: 9216 -> 6200
        fc2: 6200 -> 3200
        fc3: 3200 -> 200
        """
        super(AlexNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(192)

        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.flatten = nn.Flatten()

        # Slightly different FC sizes from original AlexNet
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 200)

        # AlexNet-style dropout in classifier
        self.dropout = nn.Dropout(p=0.5)






    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(self.relu(self.bn5(self.conv5(x))))

        x = self.flatten(x)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x