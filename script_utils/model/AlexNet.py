
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        num_classes = 10

        self.sequential = nn.Sequential(
              nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
              nn.ReLU(),
              nn.BatchNorm2d(64),
              nn.MaxPool2d(kernel_size=2),
              nn.Conv2d(64, 96, kernel_size=3, padding=2),
              nn.ReLU(),
              nn.BatchNorm2d(96),
              nn.MaxPool2d(kernel_size=2),
              nn.Conv2d(96, 96, kernel_size=3, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(96),
              nn.Conv2d(96, 64, kernel_size=3, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(64),
              nn.Conv2d(64, 64, kernel_size=3, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(64),
              nn.MaxPool2d(kernel_size=3, stride=2),
              nn.Flatten(),
              nn.Linear(1024, 128),
              nn.ReLU(),
              nn.Linear(128, 256),
              nn.ReLU(),
              nn.Linear(256, num_classes),
          )