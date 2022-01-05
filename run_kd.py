import os
import random
from typing import Mapping

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils import data
from torchsummary import summary

from kd.student import StudentMNIST
from kd.teacher import TeacherMNIST

random_seed = 42

# set random seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

lr = 1e-3  # learning rate
epoch = 5  # number of epochs


def get_dataloader(phase="train"):
    # prepare data
    transform = transforms.Compose(
        [transforms.ToTensor()])

    batch_size = 128

    if phase == "train":
        trainset = torchvision.datasets.MNIST(
            "./data", download=True, train=True, transform=transform)

        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)

    else:
        testset = torchvision.datasets.MNIST(
            "./data", download=True, train=False, transform=transform)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)

    return dataloader


# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to train teacher and student from scratch


def train(model_class, dataloader, epochs, save_loc, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    model = model_class()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            model.train()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device).view(-1, 28 * 28)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('[%d, %5d] training loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        print("test acc: ", acc(model, get_dataloader(phase="test")))
        running_loss = 0.0

    torch.save(model.state_dict(), save_loc)

    print('Finished Training')
    print('=' * 50)


# function to train distillation
def train_kd(student_cls, teacher_cls, teacher_state_pth, dataloader, epochs, save_loc, temp=1, lmbda=0.5):  # for distillation temp>0
    if os.path.exists(save_loc):
        os.remove(save_loc)

    teacher = teacher_cls().to(device)
    student = student_cls().to(device)
    teacher.load_state_dict(torch.load(teacher_state_pth))

    teacher.eval()
    student.train()

    criterion = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    optimizer = optim.Adam(student.parameters(), lr=lr)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_soft_loss = 0.0
        running_hard_loss = 0.0
        student.train()

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device).view(-1, 28 * 28)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            student_outputs = student(inputs)

            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs_soft = F.log_softmax(student_outputs/temp, dim=1)
            teacher_outputs_soft = F.log_softmax(teacher_outputs/temp, dim=1)

            hard_loss = criterion(student_outputs, labels)
            soft_loss = kd_loss(student_outputs_soft, teacher_outputs_soft)

            loss = (1 - lmbda) * soft_loss + lmbda * hard_loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_soft_loss += soft_loss.item()
            running_hard_loss += hard_loss.item()

        print('[%d, %5d] training loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        print("test acc: ", acc(student, get_dataloader(phase="test")))
        running_loss = running_soft_loss = running_hard_loss = 0.0
    torch.save(student.state_dict(), save_loc)

    print('Finished Training')


# function to measure accuracy
def acc(model, dataloader):
    model.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = round(100 * (correct/total), 2)
    return acc


if __name__ == '__main__':
    print("Teacher Network")
    summary(TeacherMNIST().to(device), input_size=(1, 784))

    print("")

    print("Student Network")
    summary(StudentMNIST().to(device), input_size=(1, 784))

    print("Training teacher")
    train(TeacherMNIST, get_dataloader(phase="train"), epoch, "teacher.pth")

    print("Training student without distillation")
    train(StudentMNIST, get_dataloader(phase="train"),
          epoch, "student_from_scratch.pth")

    print("Training student with distillation")
    train_kd(StudentMNIST, TeacherMNIST, "teacher.pth", get_dataloader(phase="train"),
             epoch, "student_distil.pth", temp=10, lmbda=0.1)

    teacher = TeacherMNIST().to(device)
    teacher.load_state_dict(torch.load("teacher.pth"))

    student_scratch = StudentMNIST().to(device)
    student_scratch.load_state_dict(
        torch.load("student_from_scratch.pth"))

    student_distil = StudentMNIST().to(device)
    student_distil.load_state_dict(
        torch.load("student_distil.pth"))

    # accuracy of teacher"
    print(
        f'TeacherMNIST accuracy: {acc(teacher, get_dataloader(phase="test"))}')

    # accuracy of teacher"
    print(
        f'StudentMNIST accuracy without distillation: {acc(student_scratch, get_dataloader(phase="test"))}')

    # accuracy of teacher"
    print(
        f'StudentMNIST accuracy with distillation: {acc(student_distil, get_dataloader(phase="test"))}')
