"""
Build a simple neural networks that takes as input a vector of size 200*3 and outputs a vector of size 200 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.SiameseTrainingUtils import PairDatasetPointNet2, SiameseNetwork, train_siamese_model, test_siamese_model
from torch_geometric.loader import DataLoader


class PointCloudEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        input_dim = num_points * 3  # since each point has x, y, z

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100)  # final output
        )

    def forward(self, x):
        # x: (batch_size, N, 3)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # flatten to (batch_size, N*3)
        return self.net(x)
    
if __name__ == "__main__":
    # import augmented dataset 
    # Set device to GPU if available, else fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = "data/aligned_brains_point_clouds_augmented/train_dataset.pt"
    test_path = "data/aligned_brains_point_clouds_augmented/test_dataset.pt"
    # code used to load the augmented dataset from directory
    print(f"Loading dataset from {train_path}")
    train_dataset = PairDatasetPointNet2.load_pointnet2_dataset(root = "./data/run_data/train", path = train_path, device = device)
    print(f"Loading dataset from {test_path}")
    test_dataset = PairDatasetPointNet2.load_pointnet2_dataset(root = "./data/run_data/train", path = test_path, device = device)

    # check that the datasets have been loaded correctly
    print("Checking datasets...")
    print(f"len(train_dataset.data_list) = {len(train_dataset.data_list)}")
    print(f"len(test_dataset.data_list) = {len(test_dataset.data_list)}")


    # Define DataLoader(s)
    print(f"Defining DataLoader(s)...")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    print(f"train loader batch size = {train_loader.batch_size}")

    print(f"train loader len = {len(train_loader)}")
    print(f"test loader len = {len(test_loader)}")

    criterion = torch.nn.MSELoss()


    # This is our model with 100 output features
    model = PointCloudEncoder(num_points=200)
    siamese_model = SiameseNetwork(core_model=model)
    # Move model to device (GPU or CPU)
    siamese_model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.001)
    # Call the training function
    train_siamese_model(siamese_model, train_loader, optimizer, criterion, device=device, epochs=20, checkpoint_interval=3)
    
    print(f"Testing has started...")
    test_siamese_model(siamese_model, test_loader, criterion, device=device, save_path = "naive_mlp/predictions.csv")
