import sys
sys.path.append("./lib/Pointcept/pointcept/models/point_transformer")
import point_transformer_regression as pt
import torch
from utils.SiameseTrainingUtils import PairDatasetPointNet2, SiameseNetwork
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the device IDs for DataParallel (use all available GPUs)
device_ids = [0, 1]  # Assuming you want to use GPU 0 and GPU 1
# set batch size
batch_size = 64

train_path = "data/aligned_brains_point_clouds_augmented/train_dataset.pt"
test_path = "data/aligned_brains_point_clouds_augmented/test_dataset.pt"

# Load the augmented dataset from directory
print(f"Loading dataset from {train_path}")
train_dataset = PairDatasetPointNet2.load_pointnet2_dataset(root="./data/run_data/train", path=train_path, device=device)
print(f"Loading dataset from {test_path}")
test_dataset = PairDatasetPointNet2.load_pointnet2_dataset(root="./data/run_data/train", path=test_path, device=device)

# Check that the datasets have been loaded correctly
print("Checking datasets...")
print(f"len(train_dataset.data_list) = {len(train_dataset.data_list)}")
print(f"len(test_dataset.data_list) = {len(test_dataset.data_list)}")

# Define DataLoader(s)
print(f"Defining DataLoader(s)...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"train loader batch size = {train_loader.batch_size}")
print(f"train loader len = {len(train_loader)}")
print(f"test loader len = {len(test_loader)}")

# Define the loss function
criterion = torch.nn.MSELoss()

# Define the model
model = pt.pointTrasformerRegression(in_channels = 3, output_size = 100)
siamese_model = SiameseNetwork(core_model=model)
#siamese_model = load_siamese_pointnet2_model(device=device, path = path)
# Move model to device (GPU or CPU)
siamese_model.to(device)

# Wrap the model with DataParallel
siamese_model = torch.nn.DataParallel(siamese_model, device_ids=device_ids)

# Define the optimizer
optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.001)

# Call the training function
print(f"Starting training...")
train_pn2_model(siamese_model, train_loader, optimizer, criterion, device=device, epochs=20, checkpoint_interval=1)

    # Call the testing function
print(f"Testing has started...")
test_pn2_model(siamese_model, test_loader, criterion, device=device)
print(f"Testing has finished...")
