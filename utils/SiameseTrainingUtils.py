from torch_geometric.data import Dataset
from augmentation_utils import augment_point_clouds_batch 
from PreProcessing import load_distance_matrix, load_data
import torch
import torch.nn as nn
import csv
from tqdm import tqdm
import sys
sys.path.append("./lib/Pointnet_Pointnet2_pytorch/models")
import pointnet2_regression_msg as pn2



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
    
    def save_to_file(self, path):
        torch.save({
            'data_list': self.data_list,
            'target_matrix': self.target_matrix
        }, path)

class PairDatasetPointNet2(PairDataset):
    def __init__(self, root, data_list, target_matrix, transform=None, pre_transform=None, pre_filter=None, augmentation = 0, device = 'cpu'):
        print(f"Instantiating PairDatasetPointNet2 with augmentation parameter {augmentation}")
        augmented_dataset = data_list
        if augmentation > 0:
            print(f"Augmenting dataset...")
            for _ in range(augmentation):
                temp = augment_point_clouds_batch(data_list.copy(), device)
                augmented_dataset = augmented_dataset + temp
            print(f"Augmentation finished.")
        super().__init__(root, augmented_dataset, target_matrix, transform=None, pre_transform=None, pre_filter=None)
    
    @staticmethod
    def load_pointnet2_dataset(root, path, transform=None, pre_transform=None, pre_filter=None, device = 'cpu'):
        loaded = torch.load(path, weights_only=False)
        data_list = loaded['data_list']
        target_matrix = loaded['target_matrix']
        return PairDatasetPointNet2(root, data_list, target_matrix, device=device)

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

def train_siamese_model(siamese_model, train_loader, optimizer, criterion, device=torch.device('cpu'), epochs=10, checkpoint_interval=5):
    for epoch in range(epochs):
        siamese_model.train()
        running_loss = 0.0

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        batch_iterator = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc="Training")

        for i, (data1, data2, target) in batch_iterator:
            optimizer.zero_grad()
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)

            output = siamese_model(data1, data2)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_iterator.set_postfix(loss=running_loss / i)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss}")
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(siamese_model, optimizer, epoch + 1, avg_loss, checkpoint_path=f"./naive_mlp/checkpoint_epoch_{epoch + 1}.pth")


def train_pn2_model(siamese_model, train_loader, optimizer, criterion, device, epochs=10, checkpoint_interval=5):
    for epoch in range(epochs):
        siamese_model.train()  # Set model to training mode
        running_loss = 0.0

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        batch_iterator = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc="Training")

        for i, (data1, data2, target) in batch_iterator:
            optimizer.zero_grad()  # Zero gradients
            
            # Move data to the primary device (e.g., cuda:0)
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            
            # Forward pass through the model
            output = siamese_model(data1.permute(0, 2, 1), data2.permute(0, 2, 1))
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            batch_iterator.set_postfix(loss=running_loss / i)

        # Compute average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")

        # Save checkpoint at specified intervals
        if (epoch + 1) % checkpoint_interval == 0:
            # Use .module to access the underlying model if wrapped with DataParallel
            save_checkpoint(
                siamese_model.module if hasattr(siamese_model, "module") else siamese_model,
                optimizer,
                epoch + 1,
                avg_loss,
                checkpoint_path=f"./checkpoint_epoch_{epoch + 1}.pth"
            )

def train_pn2_model_single_epoch(siamese_model, train_loader, optimizer, criterion, device, epoch):
    siamese_model.train()  # Set model to training mode
    running_loss = 0.0

    print(f"\nEpoch [{epoch}]")
    batch_iterator = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc="Training")

    for i, (data1, data2, target) in batch_iterator:
        optimizer.zero_grad()  # Zero gradients
            
        # Move data to the primary device (e.g., cuda:0)
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            
        # Forward pass through the model
        output = siamese_model(data1.permute(0, 2, 1), data2.permute(0, 2, 1))
            
        # Calculate loss
        loss = criterion(output, target)
            
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
            
        # Accumulate loss
        running_loss += loss.item()
        batch_iterator.set_postfix(loss=running_loss / i)

    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}] Avg Loss: {avg_loss:.4f}")

    # Save checkpoint at specified intervals
    
    # Use .module to access the underlying model if wrapped with DataParallel
    save_checkpoint(
        siamese_model.module if hasattr(siamese_model, "module") else siamese_model,
        optimizer,
        epoch,
        avg_loss,
        checkpoint_path=f"./checkpoint_epoch_{epoch}.pth"
    )

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


def test_siamese_model(model, test_loader, criterion, save_path='predictions.csv', device = torch.device('cpu')):
    model.eval()
    total_loss = 0.0
    predictions = []

    with torch.no_grad():
        for data1, data2, target in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
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

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoint.pth"):
    # Save model weights and optimizer state
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    # Load model weights and optimizer state
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}")
    return model, optimizer, epoch, loss

def load_siamese_model_checkpoint(checkpoint_path):
    model = SiameseNetwork(core_model=pn2.get_model(num_class=100, normal_channel=False)) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, optimizer, epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)
    return model, optimizer, epoch, loss



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    data_directory = "./data/aligned_brains_point_clouds"
    distance_matrix_path = "./data/phylo_trees/allspeciesList_distmat.txt"
    print(f"Loading distance matrix...")
    target = load_distance_matrix(distance_matrix_path)
    print(f"Loading point clouds...")
    data_list = load_data(data_directory)
    data = data_list[0]
    # do a random rotation 
    rotated = [data].copy()
    augment_point_clouds_batch(rotated, device = 'cpu')
    rotated = rotated[0]
    # save img of point clouds 
    pc = data.x
    pc_rotated = rotated.x
    # convert to numpy
    pc1 = pc.cpu().numpy()
    pc2 = pc_rotated.cpu().numpy()
    # save img of point clouds
    # Plot and save Brain 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='red', s=5)
    ax.set_title("Brain 1")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("brain1.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot and save Brain 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='blue', s=5)
    ax.set_title("Brain 2")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("brain2.png", dpi=300, bbox_inches='tight')
    plt.close()