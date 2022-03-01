import torch
from torch._C import dtype
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch_points3d.modules.GMM.rotation_gmm import RotationGMMConv

from visualize import show_points
from utils.harmonic import mask_idx


class GMMNet(torch.nn.Module):
    def __init__(self,
                 n_neighbours,
                 n_classes,
                 hidden,
                 * args,
                 **kwargs):
        super(GMMNet, self).__init__()
        self.n_neighbours = n_neighbours
        self.fc0 = nn.Linear(3, 16)
        self.conv1 = RotationGMMConv(16, 32, dim=2, kernel_size=25)
        self.conv2 = RotationGMMConv(32, 64, dim=2, kernel_size=25)
        self.conv3 = RotationGMMConv(64, 128, dim=2, kernel_size=25)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, data):
        edge_index = data.neighbours
        edge_attr = data.maps
        x = data.x
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr)).float()
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
