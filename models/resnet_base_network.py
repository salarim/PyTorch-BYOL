import torch
from models.resnet import resnet18, resnet50
from models.mlp_head import MLPHead


class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = resnet18(pretrained=False, num_classes=10, high_resolusion=False)
        elif kwargs['name'] == 'resnet50':
            resnet = resnet50(pretrained=False, num_classes=10, high_resolusion=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
