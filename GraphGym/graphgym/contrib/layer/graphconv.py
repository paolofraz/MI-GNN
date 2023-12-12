import torch
import torch.nn as nn
from torch.nn import Parameter
import torch_geometric.nn.conv as conv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from graphgym.config import cfg
from graphgym.register import register_layer

# Note: A registered GNN layer should take 'batch' as input
# and 'batch' as output

# Example 2: First define a PyG format Conv layer
# Then wrap it to become GraphGym format
class GraphConvLayer(MessagePassing):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GraphConvLayer, self).__init__()
        self.model = conv.GraphConv(dim_in, dim_out) # Default bias and "sum" aggregation
        self.model.reset_parameters()

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


# Remember to register your layer!
register_layer('graphconv', GraphConvLayer)
