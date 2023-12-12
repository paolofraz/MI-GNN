from torch_scatter import scatter

from graphgym.register import register_pooling
from torch_geometric.nn.aggr import MultiAggregation


def global_example_pool(x, batch, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


register_pooling('example', global_example_pool)

@register_pooling('multiagg')
def multi_aggregation(x, batch):
    multi_aggr = MultiAggregation(aggrs=['mean', 'sum', 'max'])
    return multi_aggr(x, batch)

