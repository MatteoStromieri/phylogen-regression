import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch.utils.data import Dataset
from PreProcessing import PointCloud, pack_clouds, load_point_clouds, load_data, load_distance_matrix, load_common_to_species
#from siamese_network import SiameseGNN 
from torch.utils.data import random_split
import csv
import os
from sklearn.model_selection import train_test_split
import seaborn as sns


def test_siamese_network_save_results(model, test_loader, criterion, save_path='predictions.csv'):
    model.eval()
    total_loss = 0.0
    predictions = []

    with torch.no_grad():
        for data1, data2, target in test_loader:
            output = model(data1, data2)
            loss = criterion(output, target)
            total_loss += loss.item()

            # Salva coppie di valori (predetto, reale)
            for pred, real in zip(output.cpu().numpy(), target.cpu().numpy()):
                predictions.append([float(pred), float(real)])

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    # Salva su CSV
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Predicted', 'Target'])
        writer.writerows(predictions)

    print(f"Predictions saved to {save_path}")
    return avg_loss


# training code for Siamese Network 
"""
data_directory = "./data/aligned_brains_knn_graph"
distance_matrix_path = "data/phylo_trees/allspeciesList_distmat.txt"
target = load_distance_matrix(distance_matrix_path)
data_list = load_data(data_directory)

print(f"Splitting dataset...")
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# Crea i dataset di coppie
train_dataset = PairDataset("./data/run_data/train", data_list = train_data, target_matrix = target)
test_dataset = PairDataset("./data/run_data/test", data_list = train_data, target_matrix = target)
print(f"Defining DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f"Instantiating model...")
model = SiameseGNN(in_channels=1, hidden_channels=10, out_channels=16)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

print(f"Training is starting...")
# Training del modello
train_siamese_network(model, train_loader, optimizer, criterion, epochs=10)
print(f"Testing has begun...")
test_siamese_network_save_results(model, test_loader, criterion)"
"""