import torch
import torch.nn as nn
import torch.nn.functional as F


# teacher network
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=0)
        self.conv2 = nn.Conv2d(16, 48, 3, padding=0)
        self.conv3 = nn.Conv2d(48, 128, 3, padding=0)
        # self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
#     self.conv1d1 = nn.Conv2d(128, 256, 1, padding=0)
#     self.conv1d2 = nn.Conv2d(512, 256, 1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        # print(x.shape)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TeacherMNIST(nn.Module):
    def __init__(self, in_features=784, num_classes=10):
        super(TeacherMNIST, self).__init__()

        self.layer1 = nn.Linear(in_features, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)

        return x
