import torch.nn as nn
from src.models.resnet18 import Resnet18

class Identity(nn.Module):
    def forward(self, x): return x


def resnet18_backbone():
    m = Resnet18(classes=100)
    if(hasattr(m, "fc")):
        m.fc = Identity()
    return m

