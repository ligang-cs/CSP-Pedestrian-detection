import torch
import torch.nn as nn

from lib.dla_up import dla34up

class DLA_34(nn.Module):
    def __init__(self):
        super(DLA_34, self).__init__()

        self.base_DLA = dla34up()
        self.feat = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_bn = nn.BatchNorm2d(256, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_normal_(self.feat.weight)
        # self.base_DLA = get_DLA_net()
        # self.DLA_head = nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=True)
        # self.feat_bn = nn.BatchNorm2d(256, momentum=0.01)

    # def init_weights(self):
    #     nn.init.xavier_normal_(self.DLA_head.weight)
    #     nn.init.constant_(self.DLA_head.bias, 0)

    def forward(self, x):
        feat = self.base_DLA(x)
        feat = self.feat(feat)
        feat = self.relu(self.feat_bn(feat))
        return feat