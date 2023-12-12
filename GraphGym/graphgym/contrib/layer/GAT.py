import torch
import torch.nn as nn
from torch.nn import Parameter
import torch_geometric.nn.conv as conv
from torch_geometric.nn.conv import MessagePassing

from graphgym.config import cfg
from graphgym.register import register_layer


# Note: A registered GNN layer should take 'batch' as input
# and 'batch' as output

# Example 2: First define a PyG format Conv layer
# Then wrap it to become GraphGym format
class GATConvLayer(MessagePassing):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GATConvLayer, self).__init__()
        self.model = conv.GATConv(dim_in, dim_out)  # Default heads and "concat" aggregation
        self.model.reset_parameters()

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


# Remember to register your layer!
register_layer('GATConv', GATConvLayer)
