import torch
from torch.nn import Parameter
from torch_geometric.nn import global_max_pool, global_mean_pool, MessagePassing
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric.transforms as T
import torch.nn.functional as F

from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.core.common_modules.spatial_transform import BaseLinearTransformSTNkD
from torch_points3d.models.base_model import BaseInternalLossModule
from transforms.scale_mask import ScaleMask
from transforms.harmonic_precomp import HarmonicPrecomp

EPS = 1e-12


class HarmonicNet(torch.nn.Module):
    def __init__(self,
                 nf,
                 n_classes,
                 max_order,
                 n_rings,
                 radii,
                 batch_size,
                 * args,
                 **kwargs):
        super(HarmonicNet, self).__init__()
        self.max_order = max_order
        self.n_rings = n_rings
        self.radii = radii
        self.batch_size = batch_size

        # Final Harmonic Convolution
        # We set offset to False,
        # because we will only use the radial component of the features after this
        self.conv_final = HarmonicConv(
            nf[0], n_classes, max_order, n_rings, offset=False)

        self.bias = nn.Parameter(torch.Tensor(n_classes))
        zeros(self.bias)

    def forward(self, data):
        print(data)
        # The input x is fed to our convolutional layers as a complex number and organized by rotation orders.
        # Resulting matrix: [batch_size, max_order + 1, channels, complex]
        x = torch.stack((data.x, torch.zeros_like(data.x)),
                        dim=-1).unsqueeze(1)
        batch_size = data.num_graphs
        n_nodes = x.size(0)

        # Block 1, scale 0
        # Mask correct edges and nodes
        data_scale0 = scale0_transform(data)
        # Get edge indices and precomputations for scale 0
        attributes = (data_scale0.edge_index,
                      data_scale0.precomp, data_scale0.connection)

        # Apply convolutions
        x = self.conv_final(x, *attributes)
        # Take radial component of each complex feature
        x = magnitudes(x, keepdim=False)
        # Sum the two streams
        x = x.sum(dim=1)

        x = x + self.bias
        return F.log_softmax(x, dim=1)


### Complex functions ###


def complex_product(a_re, a_im, b_re, b_im):
    """
    Computes the complex product of a and b, given the real and imaginary components of both.
    :param a_re: real component of a
    :param a_im: imaginary component of a
    :param b_re: real component of a
    :param b_im: imaginary component of a
    :return: tuple of real and imaginary components of result
    """
    a_re_ = a_re * b_re - a_im * b_im
    a_im = a_re * b_im + a_im * b_re
    return a_re_, a_im


def magnitudes(x, eps=EPS, keepdim=True):
    """
    Computes the magnitudes of complex activations.
    :param x: the complex activations.
    :param eps: offset to add, to overcome zero gradient.
    :param keepdim: whether to keep the dimensions of the input.
    """
    r = torch.sum(x * x, dim=-1, keepdim=keepdim)
    eps = torch.ones_like(r) * eps
    return torch.sqrt(torch.max(r, eps))


class HarmonicConv(MessagePassing):
    def __init__(self, in_channels, out_channels, max_order=1, n_rings=2, prev_order=1,
                 offset=True, separate_streams=True):
        super(HarmonicConv, self).__init__(
            aggr='add', flow='target_to_source', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prev_order = prev_order
        self.max_order = max_order
        self.n_rings = n_rings
        self.offset = offset
        self.separate_streams = separate_streams

        n_orders = (prev_order + 1) * (max_order +
                                       1) if separate_streams else (max_order + 1)
        self.radial_profile = Parameter(torch.Tensor(
            n_orders, n_rings, out_channels, in_channels))

        if offset:
            self.phase_offset = Parameter(torch.Tensor(
                n_orders, out_channels, in_channels))
        else:
            self.register_parameter('phase_offset', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.radial_profile)
        glorot(self.phase_offset)

    def forward(self, x, edge_index, precomp, connection=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        assert connection is None or connection.size(1) == 2
        assert precomp.dim() == 4

        out = self.propagate(edge_index=edge_index, x=x,
                             precomp=precomp, connection=connection)

        return out

    def message(self, x_j, precomp, connection):
        """
        Locally aligns features with parallel transport (using connection) and
        applies the precomputed component of the circular harmonic filter to each neighbouring node (the target nodes).
        :param x_j: the feature vector of the target neighbours [n_edges, prev_order + 1, in_channels, 2]
        :param precomp: the precomputed part of harmonic networks [n_edges, max_order + 1, n_rings, 2].
        :param connection: the connection encoding parallel transport for each edge [n_edges, 2].
        :return: the message from each target to the source nodes [n_edges, n_rings, in_channels, prev_order + 1, max_order + 1, 2]
        """

        (N, M, F, C), R = x_j.size(), self.n_rings

        # Set up result tensors
        res = torch.cuda.FloatTensor(
            N, R, F, M, self.max_order + 1, C).fill_(0)

        # Compute the convolutions per stream
        for input_order in range(M):
            # Fetch correct input order and reshape for matrix multiplications
            x_j_m = x_j[:, input_order, None, :, :]  # [N, 1, in_channels, 2]

            # First apply parallel transport
            if connection is not None and input_order > 0:
                rot_re = connection[:, None, None, 0]
                rot_im = connection[:, None, None, 1]
                x_j_m[..., 0], x_j_m[..., 1] = complex_product(
                    x_j_m[..., 0], x_j_m[..., 1], rot_re, rot_im)

            # Next, apply precomputed component
            for output_order in range(self.max_order + 1):
                m = output_order - input_order
                sign = np.sign(m)
                m = np.abs(m)

                # Compute product with precomputed component
                res[:, :, :, input_order, output_order, 0], res[:, :, :, input_order, output_order, 1] = complex_product(
                    x_j_m[..., 0], x_j_m[..., 1],
                    precomp[:, m, :, 0, None], sign * precomp[:, m, :, 1, None])

        return res

    def update(self, aggr_out):
        """
        Updates node embeddings with circular harmonic filters.
        This is done separately for each rotation order stream.

        :param aggr_out: the result of the aggregation operation [n_nodes, n_rings, in_channels, prev_order + 1, max_order + 1, complex]
        :return: the new feature vector for x [n_nodes, max_order + 1, out_channels, complex]
        """
        (N, _, F, M, _, C), O = aggr_out.size(), self.out_channels
        res = torch.cuda.FloatTensor(N, M, self.max_order + 1, O, 2).fill_(0)

        for input_order in range(M):
            for output_order in range(self.max_order + 1):
                m = np.abs(output_order - input_order)
                m_idx = input_order * (self.max_order + 1) + \
                    output_order if self.separate_streams else m

                # [N, n_rings, 1, in_channels]
                aggr_re = aggr_out[:, :, None, :, input_order, output_order, 0]
                # [N, n_rings, 1, in_channels]
                aggr_im = aggr_out[:, :, None, :, input_order, output_order, 1]

                # Apply the radial profile
                # [N, out_channels, in_channels]
                aggr_re = (self.radial_profile[m_idx] * aggr_re).sum(dim=1)
                # [N, out_channels, in_channels]
                aggr_im = (self.radial_profile[m_idx] * aggr_im).sum(dim=1)

                # Apply phase offset
                if self.offset:
                    # [out_channels, in_channels]
                    cos = torch.cos(self.phase_offset[m_idx])
                    # [out_channels, in_channels]
                    sin = torch.sin(self.phase_offset[m_idx])
                    aggr_re, aggr_im = complex_product(
                        aggr_re, aggr_im, cos, sin)

                # Store per rotation stream
                res[:, input_order, output_order, :, 0] = aggr_re.sum(dim=-1)
                res[:, input_order, output_order, :, 1] = aggr_im.sum(dim=-1)

        # The input streams are summed together to retrieve one value per output stream
        return res.sum(dim=1)

    def forward_embedding(self, pos, batch):
        global_feat, local_feat = self.forward(pos, batch)
        indices = batch.unsqueeze(-1).repeat((1, global_feat.shape[-1]))
        gathered_global_feat = torch.gather(global_feat, 0, indices)
        x = torch.cat([local_feat, gathered_global_feat], -1)
        return x
