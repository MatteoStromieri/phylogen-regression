import sys
sys.path.append("./lib/Pointnet_Pointnet2_pytorch/models")
import pointnet2_regression_msg as pn2
from utils.PreProcessing import PointCloud, pack_clouds, load_point_clouds, load_data, load_distance_matrix, load_common_to_species
from sklearn.model_selection import train_test_split
from utils.SiameseTrainingUtils import PairDatasetPointNet2, SiameseNetwork, train_pn2_model, test_pn2_model, train_pn2_model_single_epoch, load_siamese_model_checkpoint
from torch_geometric.loader import DataLoader
import torch

def generate_and_save_dataset(device):
    data_directory = "./data/aligned_brains_point_clouds"
    distance_matrix_path = "./data/phylo_trees/allspeciesList_distmat.txt"
    print(f"Loading distance matrix...")
    target = load_distance_matrix(distance_matrix_path)
    print(f"Loading point clouds...")
    data_list = load_data(data_directory)

    # code used to generate the augmented dataset 

    print(f"Splitting dataset...")
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=30)
    print(f"len(train_data) = {len(train_data)})")
    print(f"len(test_data) = {len(test_data)})")
    # Create the dataset of pairs
    print(f"Defining PairDataset(s)...")
    train_dataset = PairDatasetPointNet2("./data/run_data/train", data_list=train_data, target_matrix=target, augmentation=4, device=device)
    test_dataset = PairDatasetPointNet2("./data/run_data/test", data_list=test_data, target_matrix=target, augmentation=4, device=device)

    # save augmented dataset
    train_dataset.save_to_file("data/aligned_brains_point_clouds_augmented/train_dataset.pt")
    test_dataset.save_to_file("data/aligned_brains_point_clouds_augmented/test_dataset.pt")





def data_parallel_main(batch_size):
    # Set device to GPU if available, else fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the device IDs for DataParallel (use all available GPUs)
    device_ids = [0, 1]  # Assuming you want to use GPU 0 and GPU 1

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
    model = pn2.get_model(num_class=100, normal_channel=False)
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


def data_parallel_epoch(batch_size, epoch):
    # Set device to GPU if available, else fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the device IDs for DataParallel (use all available GPUs)
    device_ids = [0, 1]  # Assuming you want to use GPU 0 and GPU 1

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

    checkpoint_path = f"checkpoint_epoch_{epoch-1}.pth"
    # Define the loss function
    criterion = torch.nn.MSELoss()
    siamese_model, optimizer, _, _ = load_siamese_model_checkpoint(checkpoint_path)
    
    siamese_model.to(device)

    # Wrap the model with DataParallel
    siamese_model = torch.nn.DataParallel(siamese_model, device_ids=device_ids)

    # Call the training function
    print(f"Starting training...")
    train_pn2_model_single_epoch(siamese_model, train_loader, optimizer, criterion, device=device, epoch=epoch)

    print(f"Epoch {epoch} has finished training")


def single_gpu_training():
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
    model = pn2.get_model(num_class=100, normal_channel=False)
    siamese_model = SiameseNetwork(core_model=model)
    # Move model to device (GPU or CPU)
    siamese_model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.001)
    # Call the training function
    train_pn2_model(siamese_model, train_loader, optimizer, criterion, device=device, epochs=20)
    
    print(f"Testing has started...")
    test_pn2_model(siamese_model, test_loader, criterion, device=device)

if __name__=="__main__":
    # Convert the argument to an integer
    generate_and_save_dataset(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
    #data_parallel_main(batch_size=250)
