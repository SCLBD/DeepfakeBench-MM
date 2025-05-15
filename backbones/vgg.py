import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch import nn
from einops import rearrange

from utils.registry import BACKBONE

@BACKBONE.register_module(module_name="vgg")
class VGG(nn.Module):
    def __init__(self, last_dim=256, last_avg=False, temporal_half=False, **kwargs):
        super(VGG, self).__init__()
        self.last_avg = last_avg

        num_filters = [last_dim // 8, last_dim // 4, last_dim // 2, last_dim]

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=(2, 1)),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=(1, 5), padding=0),
            nn.Flatten(2, 3),
        )

        # if self.last_avg:
        #     self.mlp_head = nn.Sequential(
        #         nn.Flatten(),
        #         nn.ReLU(),
        #         nn.Linear(last_dim, last_dim),
        #     )
        # else:
        #     self.mlp_head = nn.Sequential(
        #         nn.ReLU(),
        #         nn.Linear(last_dim, last_dim),
        #     )

        if temporal_half:
            self.temporal_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)
        else:
            self.temporal_pool = nn.Identity()
    def features(self, x):
        pass

    def end_points(self, x):
        pass

    def classifier(self, x):
        pass

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = x.transpose(1, 2)
        x = rearrange(x, 'b c t -> b t c')
        x = self.temporal_pool(x)
        if self.last_avg:
            x = torch.mean(x, 1)
        # x = self.mlp_head(x)

        return x

