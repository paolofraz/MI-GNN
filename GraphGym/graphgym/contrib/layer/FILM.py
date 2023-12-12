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
class FiLMConvLayer(MessagePassing):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(FiLMConvLayer, self).__init__()
        self.model = conv.FiLMConv(dim_in, dim_out)
        self.model.reset_parameters()

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


# Remember to register your layer!
register_layer('FiLMConv', FiLMConvLayer)
