from GNNConvWeighted import GCNConvWeighted
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.data import Dataset
import random 


class SiameseGNN(nn.Module):
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

    def forward(self, data1, data2):
        # Unpack data1
        x1, edge_index1, edge_attr1, batch1 = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        # Unpack data2
        x2, edge_index2, edge_attr2, batch2 = data2.x, data2.edge_index, data2.edge_attr, data2.batch
        
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
        

        # Run the architecture on the second graph 
        x2 = self.conv1(x2, edge_index2, edge_attr2)
        x2 = self.activation1(x2)
        x2 = self.conv2(x2, edge_index2, edge_attr2)
        x2 = self.activation1(x2)
        x2 = self.conv3(x2, edge_index2, edge_attr2)
        x2 = self.conv4(x2, edge_index2, edge_attr2)
        x2 = self.activation1(x2)
        x2 = self.conv5(x2, edge_index2, edge_attr2)
        x2 = self.activation1(x2)
        x2 = self.conv6(x2, edge_index2, edge_attr2)
        x2 = self.activation1(x2)
        x2 = global_mean_pool(x2, batch2)
        x2 = self.fc1(x2)
        x2 = self.activation2(x2)
        x2 = self.fc2(x2)
        x2 = self.activation2(x2)
        x2 = self.fc3(x2)
        x2 = self.activation2(x2)
        x2 = self.fc4(x2)
        x2 = self.activation2(x2)
        x2 = self.fc5(x2)
        x2 = self.activation2(x2)
        x2 = self.fc6(x2)
        x2 = self.activation2(x2)

        # compute euclidean distance
        out = torch.norm(x1 - x2, p = 2, dim = -1)

        return out 

def train_siamese_network(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()  # Imposta il modello in modalità training
        running_loss = 0.0
        
        for data1, data2, target in train_loader:
            optimizer.zero_grad()  # Azzera i gradienti
            
            # Passaggio in avanti attraverso la rete
            output = model(data1, data2)
            
            # Calcolo della perdita basata sulla differenza tra l'output e il target
            loss = criterion(output, target)
            
            # Backpropagation e aggiornamento dei pesi
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()  # Aggiungi la perdita corrente
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

class PairDataset(Dataset):
    def __init__(self, root, data_list, target_matrix, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_list = data_list
        self.target_matrix = target_matrix

    def len(self):
        # Number of possible pairs (all combinations)
        return len(self.data_list) ** 2

    def get(self, idx):
        # Flatten the index space for pairs (i, j)
        n = len(self.data_list)
        i = idx // n
        j = idx % n

        data1 = self.data_list[i]
        data2 = self.data_list[j]
        label1 = data1.label
        label2 = data2.label
        target = self.target_matrix[label1, label2]

        return data1, data2, target