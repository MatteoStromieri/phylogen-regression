from torch_geometric.data import Dataset
import torch
import torch.nn as nn
import csv


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

class PairDatasetPointNet2(PairDataset):
    def get(self, idx):
        graph1, graph2, target = super().get(idx)
        # now extract node coordinates from graph1 and graph2
        xyz1 = graph1.x
        xyz2 = graph2.x
        return xyz1, xyz2, target 

class SiameseNetwork(nn.Module):
    def __init__(self, core_model, similarity_measure = lambda x1, x2 : torch.norm(x1-x2, p = 2, dim = 1)):
        super().__init__()
        self.core_model = core_model
        self.similarity_measure = similarity_measure
    
    def forward(self, input1, input2, return_embeddings=False):
        output1 = self.core_model(input1) # shape (batch_size,100)
        output2 = self.core_model(input2) # shape (batch_size,100)
        similarity = self.similarity_measure(output1, output2)
        if return_embeddings:
            return similarity, output1, output2
        return similarity
    
def train_siameseGNN_model(siamese_model, train_loader, optimizer, criterion, device=torch.device('cpu'), epochs=10):
    for epoch in range(epochs):
        siamese_model.train()  # Set model to training mode
        running_loss = 0.0
        i = 1
        
        for data1, data2, target in train_loader:
            print(f"Processing {i}-th batch of {epoch}-th epoch...")
            i += 1
            optimizer.zero_grad()  # Zero gradients
            
            # Move data to device (GPU or CPU)
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            
            # Forward pass through the model
            output = siamese_model(data1, data2)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()  # Add current loss to running loss

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

def train_pn2_model(siamese_model, train_loader, optimizer, criterion, device, epochs=10):
    for epoch in range(epochs):
        siamese_model.train()  # Set model to training mode
        running_loss = 0.0
        i = 1
        
        for data1, data2, target in train_loader:
            #print(f"Processing {i}-th batch of {epoch}-th epoch...")
            i += 1
            optimizer.zero_grad()  # Zero gradients
            
            # Move data to device (GPU or CPU)
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            
            # Forward pass through the model
            output = siamese_model(data1.permute(0,2,1), data2.permute(0,2,1))
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()  # Add current loss to running loss

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

def test_pn2_model(model, test_loader, criterion, save_path='predictions.csv', device = torch.device('cpu')):
    model.eval()
    total_loss = 0.0
    predictions = []

    with torch.no_grad():
        for data1, data2, target in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1.permute(0, 2, 1), data2.permute(0, 2, 1))
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


def test_siamese_network_save_results(model, test_loader, criterion, save_path='predictions.csv', device = torch.device('cpu')):
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
