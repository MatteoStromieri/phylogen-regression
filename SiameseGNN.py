from siamese_network.GNNConvWeighted import GCNConvWeighted
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch
import random 


class BasicGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # GNN
        self.conv1 = GCNConvWeighted(in_channels, hidden_channels)
        self.conv2 = GCNConvWeighted(hidden_channels, hidden_channels)
        self.conv3 = GCNConvWeighted(hidden_channels, hidden_channels)
        self.conv4 = GCNConvWeighted(hidden_channels, hidden_channels)
        self.conv5 = GCNConvWeighted(hidden_channels, hidden_channels)
        self.conv6 = GCNConvWeighted(hidden_channels, hidden_channels)

        # MLP
        self.fc1 = nn.Linear(hidden_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.fc3 = nn.Linear(out_channels, out_channels)
        self.fc4 = nn.Linear(out_channels, out_channels)
        self.fc5 = nn.Linear(out_channels, out_channels)
        self.fc6 = nn.Linear(out_channels, out_channels)

        # Activations 
        self.activation1 = nn.Tanh()
        self.activation2 = nn.ReLU()

    def forward(self, data):
        # Unpack data1
        x1, edge_index1, edge_attr1, batch1 = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Run the architecture on the first graph 
        x1 = self.conv1(x1, edge_index1, edge_attr1)
        x1 = self.activation1(x1)
        x1 = self.conv2(x1, edge_index1, edge_attr1)
        x1 = self.activation1(x1)
        x1 = self.conv3(x1, edge_index1, edge_attr1)
        x1 = self.conv4(x1, edge_index1, edge_attr1)
        x1 = self.activation1(x1)
        x1 = self.conv5(x1, edge_index1, edge_attr1)
        x1 = self.activation1(x1)
        x1 = self.conv6(x1, edge_index1, edge_attr1)
        x1 = self.activation1(x1)
        x1 = global_mean_pool(x1, batch1)
        x1 = self.fc1(x1)
        x1 = self.activation2(x1)
        x1 = self.fc2(x1)
        x1 = self.activation2(x1)
        x1 = self.fc3(x1)
        x1 = self.activation2(x1)
        x1 = self.fc4(x1)
        x1 = self.activation2(x1)
        x1 = self.fc5(x1)
        x1 = self.activation2(x1)
        x1 = self.fc6(x1)
        x1 = self.activation2(x1)
        
        return x1