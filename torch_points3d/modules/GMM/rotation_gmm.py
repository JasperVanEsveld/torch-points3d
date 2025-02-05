from typing import Union, Tuple
import numpy as np
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from rotation import create_rotations
from torch_points3d.modules.GMM.inits import glorot, zeros


class RotationGMMConv(MessagePassing):
    r"""The gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \frac{1}{K} \sum_{k=1}^K
        \mathbf{w}_k(\mathbf{e}_{i,j}) \odot \mathbf{\Theta}_k \mathbf{x}_j,

    where

    .. math::
        \mathbf{w}_k(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left(
        \mathbf{e} - \mathbf{\mu}_k \right)}^{\top} \Sigma_k^{-1}
        \left( \mathbf{e} - \mathbf{\mu}_k \right) \right)

    denotes a weighting function based on trainable mean vector
    :math:`\mathbf{\mu}_k` and diagonal covariance matrix
    :math:`\mathbf{\Sigma}_k`.

    .. note::

        The edge attribute :math:`\mathbf{e}_{ij}` is usually given by
        :math:`\mathbf{e}_{ij} = \mathbf{p}_j - \mathbf{p}_i`, where
        :math:`\mathbf{p}_i` denotes the position of node :math:`i` (see
        :class:`torch_geometric.transform.Cartesian`).

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int): Number of kernels :math:`K`.
        separate_gaussians (bool, optional): If set to :obj:`True`, will
            learn separate GMMs for every pair of input and output channel,
            inspired by traditional CNNs. (default: :obj:`False`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, dim: int, kernel_size: int,
                 separate_gaussians: bool = False, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super(RotationGMMConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.rel_in_channels = in_channels[0]

        self.g = Parameter(
            torch.Tensor(in_channels[0], out_channels * kernel_size))

        if not self.separate_gaussians:
            self.mu = Parameter(torch.Tensor(kernel_size, dim))
            self.sigma = Parameter(torch.Tensor(kernel_size, dim))
        else:
            self.mu = Parameter(
                torch.Tensor(in_channels[0], out_channels, kernel_size, dim))
            self.sigma = Parameter(
                torch.Tensor(in_channels[0], out_channels, kernel_size, dim))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.g)
        glorot(self.mu)
        glorot(self.sigma)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None):
        """"""
        x = x.float()
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        if not self.separate_gaussians:
            out: OptPairTensor = (torch.matmul(x[0], self.g.float()), x[1])
            out = self.propagate(edge_index, x=out, edge_attr=edge_attr,
                                 size=size)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                 size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out += torch.matmul(x_r, self.root)

        if self.bias is not None:
            out += self.bias

        return out

    def message_rotation(self, x_j: Tensor, edge_attr: Tensor):
        EPS = 1e-15
        F, M = self.rel_in_channels, self.out_channels
        (E, D), K = edge_attr.size(), self.kernel_size

        if not self.separate_gaussians:
            gaussian = -0.5 * (edge_attr.view(E, 1, D) -
                               self.mu.view(1, K, D)).pow(2)
            gaussian = gaussian / (EPS + self.sigma.view(1, K, D).pow(2))
            gaussian = torch.exp(gaussian.sum(dim=-1))  # [E, K]

            return (x_j.view(E, K, M) * gaussian.view(E, K, 1)).sum(dim=-2)

        else:
            gaussian = -0.5 * (edge_attr.view(E, 1, 1, 1, D) -
                               self.mu.view(1, F, M, K, D)).pow(2)
            gaussian = gaussian / (EPS + self.sigma.view(1, F, M, K, D).pow(2))
            gaussian = torch.exp(gaussian.sum(dim=-1))  # [E, F, M, K]

            gaussian = gaussian * self.g.view(1, F, M, K)
            gaussian = gaussian.sum(dim=-1)  # [E, F, M]

            return (x_j.view(E, F, 1) * gaussian).sum(dim=-2)  # [E, M]

    def apply_rotation(self, attr, rotation):
        result = rotation.dot(attr[0].cpu())
        rotated = np.zeros(attr.shape)
        for i, attribute in enumerate(attr.cpu()):
            rotated[i] = rotation.dot(attribute)
        return torch.tensor(rotated)

    def message(self, x_j: Tensor, edge_attr: Tensor):
        n_rotations = 8
        M = self.out_channels
        (E, _D) = edge_attr.size()
        result = np.zeros((8, E, M))
        rotations = create_rotations(n_rotations)
        for i, rotation in enumerate(rotations):
            self.apply_rotation(edge_attr, rotation)
            result[i] = self.message_rotation(
                x_j, edge_attr).cpu().detach().numpy()
        return torch.tensor(result.max(axis=0)).cuda()

    def __repr__(self):
        return '{}({}, {}, dim={})'.format(self.__class__.__name__,
                                           self.in_channels, self.out_channels,
                                           self.dim)
