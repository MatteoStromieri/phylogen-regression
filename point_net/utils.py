import os
import torch
import numpy as np

def farthest_point_sample(xyz: torch.Tensor, n_sampled_points: int) -> torch.Tensor:
    """
    :param xyz: Tensor of shape (B, N, 3) containing point cloud data, where B is the batch size and N is the number of points.
    :param npoints: Number of points to sample.
    :return: Tensor of shape (B, npoints) containing the indices of the farthest sampled points.
    """
    batch_size, num_points, _ = xyz.shape
    sampled_indices = torch.zeros(batch_size, n_sampled_points, dtype=torch.long)

    # Compute the initial point (randomly pick one point)
    dist = torch.ones(batch_size, num_points) * 1e10  # Initialize distances with a large value
    farthest_points = torch.randint(0, num_points, (batch_size, 1), dtype=torch.long)  # Random start point

    # The first farthest point for each batch is selected at random
    sampled_indices[:, 0] = farthest_points.squeeze()

    # Compute distances iteratively
    for i in range(1, n_sampled_points):
        # Compute pairwise distances from the selected points
        index = sampled_indices[:,:i].unsqueeze(2).repeat(1,1,3) # selected indices
        selected_points = xyz.gather(1, index) # selected points coordinates
        selected_points = selected_points.unsqueeze(1).repeat(1,num_points,1,1)
        temp = xyz.unsqueeze(2).repeat(1,1,i,1)
        print(temp.shape)
        print(selected_points.shape)
        norm = torch.norm(selected_points - temp, dim = 3)
        print(f"norm shape {norm.shape}")
        dist, _ = torch.min(norm, dim = 1)
        # Choose the next farthest point based on the current distances
        _, farthest_points = dist.max(dim=1, keepdim=True)
        print(f"farthest_point tensor shape {farthest_points.shape}")
        sampled_indices[:, i] = farthest_points.squeeze()

    return sampled_indices

def knn(xyz: torch.Tensor, centers: torch.Tensor, k):
    """
    :param xyz: Tensor of shape (B, N, 3) containing point cloud data, where B is the batch size and N is the number of points.
    :param centers: Tensor of shape (B, M) containing indices of point clouds centers, where b is the batch size and M is the number of selected points per batch.
    :param k: int denoting the amount of neighbors for each center
    :return: tensor of shape (B,M,k) containing indices of the k points relative to each center
    """
    B, N, _ = xyz.shape  # B: batch size, N: number of points in each cloud
    _, M = centers.shape  # M: number of centers per batch

    # Initialize tensor to store the k nearest neighbor indices
    knn_indices = torch.zeros((B, M, k), dtype=torch.long).to(xyz.device)

    # Loop through each batch and each center
    for b in range(B):
        for m in range(M):
            center_idx = centers[b, m]  # Get the index of the current center point

            # Get the 3D coordinates of the center point
            center_point = xyz[b, center_idx, :]

            # Compute the squared Euclidean distance between the center and all other points in the cloud
            distances = torch.sum((xyz[b, :, :] - center_point) ** 2, dim=1)  # Shape (N,)

            # Sort the distances and get the indices of the k nearest neighbors
            _, indices = torch.topk(distances, k + 1, largest=False)  # k+1 because the center itself is included

            # Remove the center index from the selected k nearest neighbors
            knn_indices[b, m, :] = indices[indices != center_idx][:k]

    return knn_indices

def load_point_clouds(directory):
    """
    Loads all point clouds from a directory, converts them into tensors, and
    stacks them into a single tensor.

    Parameters:
    - directory: str, path to the directory containing point cloud files

    Returns:
    - A tensor of shape (B, N, 3), where B is the number of point clouds and 
      N is the number of points in each point cloud.
    - A list of animal names (inferred from file names)
    """
    point_clouds = []
    animal_names = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.txt', '.csv', '.npy')):  # Add more extensions if needed
            path = os.path.join(directory, filename)
            
            # Load point cloud depending on format
            if filename.endswith('.npy'):
                pc = np.load(path)
            else:
                pc = np.loadtxt(path, delimiter=' ')

            # Check shape
            if pc.shape[1] != 3:
                raise ValueError(f"Point cloud in {filename} must have shape (N, 3), but got {pc.shape}")
            
            point_clouds.append(torch.tensor(pc, dtype=torch.float32))
            animal_names.append(os.path.splitext(filename)[0])  # filename without extension

    # Stack into a tensor of shape (B, N, 3)
    point_cloud_tensor = torch.stack(point_clouds)

    return point_cloud_tensor, animal_names
