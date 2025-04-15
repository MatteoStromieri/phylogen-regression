"""
We want to load the model weights for each epoch and test the model on the test dataset. Then plot the results.
"""
import sys
#sys.path.append("..")
import torch
from utils.SiameseTrainingUtils import PairDatasetPointNet2, SiameseNetwork, test_pn2_model, load_siamese_model_checkpoint
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import r2_score
import pandas as pd

def compute_r2_from_csv(pred_path, true_col='Target', pred_col='Predicted'):
    """
    Computes R² score given a CSV file with true and predicted values.

    Args:
        pred_path (str): Path to the CSV file.
        true_col (str): Column name for true values.
        pred_col (str): Column name for predicted values.

    Returns:
        float: R² score
    """
    df = pd.read_csv(pred_path)
    y_true = df[true_col].values
    y_pred = df[pred_col].values
    return r2_score(y_true, y_pred)

def plot_loss(test_dataset, batch_size, epochs):
    # Define DataLoader(s)
    print(f"Defining DataLoader(s)...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    train_loss = []
    test_loss = []

    for i in range(1,epochs+1):
        path = f"./pointnet_weights/checkpoints/checkpoint_epoch_{i}.pth"
        pred_path = f"./pointnet_weights/checkpoints/predictions_epoch_{i}.csv"
        siamese_model, _, _, loss_train = load_siamese_model_checkpoint(path)
        # Move model to device (GPU or CPU)
        siamese_model.to(device)
        # Wrap the model with DataParallel
        siamese_model = torch.nn.DataParallel(siamese_model, device_ids=device_ids)
        # Call the testing function
        print(f"[Epoch {i}]: Testing has started...")
        loss_test = test_pn2_model(siamese_model, test_loader, criterion, device=device, save_path=pred_path)
        print(f"[Epoch {i}]: Testing has finished...")
        train_loss.append(loss_train)
        test_loss.append(loss_test)
        print(f"[Epoch {i}]: Training loss = {loss_train}, Test loss = {loss_test}")
        del siamese_model
        gc.collect()
        torch.cuda.empty_cache()


    # plot the results
    epochs = list(range(1, epochs + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Test Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 

    plt.savefig("loss_plot.png")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 250
    epochs = 20

    # Set the device IDs for DataParallel (use all available GPUs)
    device_ids = [0, 1]  # Assuming you want to use GPU 0 and GPU 1

    test_path = "data/aligned_brains_point_clouds_augmented/test_dataset.pt"


    print(f"Loading dataset from {test_path}")
    test_dataset = PairDatasetPointNet2.load_pointnet2_dataset(root="./data/run_data/train", path=test_path, device=device)

    # Check that the datasets have been loaded correctly
    print("Checking datasets...")
    print(f"len(test_dataset.data_list) = {len(test_dataset.data_list)}")
    plot_loss(test_dataset, batch_size, epochs)
    print(f"Finished plotting loss...")