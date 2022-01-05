import torch
import torch.nn as nn
import torch.nn.functional as F


# student network
class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 48, 3)
#     self.conv1d = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 2 * 2, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
#     x = self.pool(F.relu(self.conv1d(x)))
#     print(x.shape)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StudentMNIST(nn.Module):
    def __init__(self, in_features=784, num_classes=10):
        super(StudentMNIST, self).__init__()

        self.layer1 = nn.Linear(in_features, 64)
        self.layer2 = nn.Linear(64, 16)
        self.layer3 = nn.Linear(16, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)

        return x
