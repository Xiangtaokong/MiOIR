import functools
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 7)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 3,3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))

    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)

        # if out.size()[2] > out.size()[3]:
        #     out = nn.AvgPool2d(out.size()[3])(out)
        # else:
        #     out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        out = F.softmax(out, dim=1)
        return out