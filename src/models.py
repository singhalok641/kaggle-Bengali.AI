import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

    
        # (last_linear): Linear(in_features=512, out_features=1000, bias=True)
        # resnet34 has 512 input features and 1000 output features
        # we will change that by adding a few extra layers

        # 168 outputs for grapheme_root
        self.l0 = nn.Linear(512, 168)   
        # 11 outputs for vowel_diacritic
        self.l1 = nn.Linear(512, 11)
        # 7 outputs for consonant diacritic
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1,l2


    




