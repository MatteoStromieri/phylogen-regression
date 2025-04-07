from torch_geometric.data import Dataset

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