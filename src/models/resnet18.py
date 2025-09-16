import torch
import torch.nn as nn
import torchvision.models as models

class Resnet18(nn.Module):
    def __init__(self, classes: int = 100, pretrained: bool = False):
        super().__init__()

        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = None
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)


        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()


        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, classes)


        self._feat = None


        def _hook(module, inp, out):
            self._feat = out.view(out.size(0), -1)

        self.backbone.avgpool.register_forward_hook(_hook)

    def forward(self, x, return_features: bool = False):
        logits = self.backbone(x)  # hook fills self._feat
        if return_features:
            return logits, self._feat
        return logits


if __name__ == "__main__":
    model = Resnet18()
    # CIFAR-sized input for realism; 224x224 also works
    x = torch.randn(1, 3, 32, 32)
    y, f = model(x, return_features=True)  # <-- correct flag name
    print("logits:", y.shape)    # torch.Size([1, 100])
    print("features:", f.shape)  # torch.Size([1, 512])
