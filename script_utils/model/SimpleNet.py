
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):

    def __init__(self):
        num_classes = 10

        self.sequential = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(True),
            nn.Maxpool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(True),
            nn.Maxpool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4 * 4 * 50, 500),
            nn.ReLU(True),
            nn.Linear(500, num_classes)
        )