import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from torch_scatter import scatter

from graphgym.config import cfg
from graphgym.register import register_layer

# Note: A registered GNN layer should take 'batch' as input
# and 'batch' as output


# Example 1: Directly define a GraphGym format Conv
# take 'batch' as input and 'batch' as output
class MIGNNv1(MessagePassing):
    r""" MIGNN - V1
    It exploits the Hadamard product between the current node projection and the aggregation of its neighbors:
    h(i)_v = (W′h(i−1)v + b′) W′′ ∑(h(i−1)u + b′′)
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MIGNNv1, self).__init__(aggr='add', **kwargs) # Use "add" aggregation for the message passing
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define learnable parameters
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, batch):
        """"""
        x, edge_index = batch.node_feature, batch.edge_index

        # Think about this
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # h = sum(W''x + b'')
        h = self.lin2(x)

        batch.node_feature = self.propagate(edge_index, x=x, h=h)

        return batch

    def message(self, h_j):
        return h_j

    def update(self, aggr_out, x):
        # out = (W'x + b') * agg_out
        return aggr_out * self.lin1(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# Remember to register your layer!
register_layer('MIGNNv1', MIGNNv1)

class MIGNNv2(MessagePassing):
    r""" MIGNN - V2
    make explicitly the combination of the additive and multiplicative building blocks.
    h(i)v = W′h(i−1) v + W′′ ∑h(i−1)u + W′′′(h(i−1)v∑hi−1u ) + b
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MIGNNv2, self).__init__(aggr='add', **kwargs) # Use "add" aggregation for the message passing
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define learnable parameters
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.W2 = Parameter(torch.Tensor(in_channels, out_channels))
        self.W3 = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        glorot(self.W2)
        glorot(self.W3)

    def forward(self, batch):
        """"""
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]
        x, edge_index = batch.node_feature, batch.edge_index

        # h = W'x + b' has shape [num_nodes, out_channels]
        h = self.lin1(x)

        batch.node_feature = self.propagate(edge_index, x=x, h=h)

        return batch

    def message(self, x_j):
        return x_j # aggr_out as previous layer

    def update(self, aggr_out, x, h):
        # out = h + W'' * agg_out + W''' (x * agg_out)
        return h + torch.matmul(aggr_out, self.W2) + torch.matmul(x * aggr_out, self.W3)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# Remember to register your layer!
register_layer('MIGNNv2', MIGNNv2)

class MIGNNv3(nn.Module):
    r""" MIGNN - V3
    exploits Hadamard products to define both the combination and aggregation steps of the GC operator
    h(k)i = (W′h(k−1)i + b′) W′′ ∏[(h(k−1)j + b′′)]
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define learnable parameters
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, batch):
        """"""
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]
        x, edge_index = batch.node_feature, batch.edge_index

        # h1 = W'x + b' has shape [num_nodes, out_channels]
        h1 = self.lin1(x)

        # Multiply h2
        #x_j = x[edge_index[0]]  # Source node features [num_edges, num_features]
        #x_i = x[edge_index[1]]  # Target node features [num_edges, num_features]

        # Aggregate messages based on target node indices
        aggr_out = scatter(x[edge_index[0]], edge_index[1], dim=0, dim_size=x.size(0), reduce='mul')

        batch.node_feature = h1 * self.lin2(aggr_out)

        return batch

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# Remember to register your layer!
register_layer('MIGNNv3', MIGNNv3)

class MIGNNv4(MessagePassing):
    r""" MIGNN - V4
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MIGNNv4, self).__init__(aggr='add', **kwargs) # Use "add" aggregation for the message passing
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.epsilon = 1e-8
        self.relu = nn.ReLU()

        # Define learnable parameters
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, batch):
        """"""
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]
        x, edge_index = batch.node_feature, batch.edge_index

        # h = W'x + b' has shape [num_nodes, out_channels]
        h1 = self.relu(self.lin1(x))

        batch.node_feature = self.propagate(edge_index, x=x, h1=h1)

        return batch

    def message(self, x_j):
        return torch.log(self.relu(x_j) + self.epsilon) # aggr_out as previous layer

    def update(self, aggr_out, h1):
        # out = h + W'' * agg_out + W''' (x * agg_out)
        return h1 * self.lin2(aggr_out)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# Remember to register your layer!
register_layer('MIGNNv4', MIGNNv4)