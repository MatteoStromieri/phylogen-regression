import torch 
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class GCNConvWeighted(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias = False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attributes):
        edge_index, edge_attributes = add_self_loops(edge_index, num_nodes=x.size(0), edge_attr=edge_attributes, fill_value=0)
        # linearly trasform the nodes feature matrix
        x = self.lin(x)
        # compute message weight 
        exp_weights = torch.exp(edge_attributes)
        # propagate messages
        out = self.propagate(edge_index, x = x, weights = exp_weights)
        # add bias
        out = out + self.bias
        return out 
    
    # x_j are source features for every edge  
    def message(self, x_j, weights):
        weights = weights.view(-1,1)
        return x_j * weights

    