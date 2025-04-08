import sys 
sys.path.append("/user/mstromie/home/code/phylogen-regression/lib/Pointnet_Pointnet2_pytorch/models")
import pointnet2_regression_msg as pn2
from PreProcessing import PointCloud, pack_clouds, load_point_clouds, load_data, load_distance_matrix, load_common_to_species
from sklearn.model_selection import train_test_split
from SiameseTrainingUtils import PairDatasetPointNet2, SiameseNetwork, train_pn2_model, test_pn2_model
from torch_geometric.loader import DataLoader
import torch

# This is our model with 100 output features
model = pn2.get_model(num_class=100, normal_channel=False)
siamese_model = SiameseNetwork(core_model=model)

# Set device to GPU if available, else fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device (GPU or CPU)
siamese_model.to(device)

data_directory = "./data/aligned_brains_point_clouds"
distance_matrix_path = "data/phylo_trees/allspeciesList_distmat.txt"
print(f"Loading distance matrix...")
target = load_distance_matrix(distance_matrix_path)
print(f"Loading point clouds...")
data_list = load_data(data_directory)

print(f"Splitting dataset...")
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# Create the dataset of pairs
print(f"Defining PairDataset(s)...")
train_dataset = PairDatasetPointNet2("./data/run_data/train", data_list=train_data, target_matrix=target)
test_dataset = PairDatasetPointNet2("./data/run_data/test", data_list=train_data, target_matrix=target)

# Define DataLoader(s)
print(f"Defining DataLoader(s)...")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print(f"train loader batch size = {train_loader.batch_size}")

# Optimizer and loss function
optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Move target (distance matrix) to the same device
target = target.to(device)

# Call the training function
train_pn2_model(siamese_model, train_loader, optimizer, criterion, device=device, epochs=1)
torch.save(siamese_model.state_dict(), './siamese_network_pointnet2/siamese_pointnet2_regression_msg_15 epochs.pth')

# Save the model's predictions after testing
save_path = "./siamese_network_pointnet2/siamese_network_pointnet2_cls_msg.csv"
test_pn2_model(siamese_model, test_loader, criterion, save_path, device=device)
