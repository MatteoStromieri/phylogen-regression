import torch 
import numpy as np

import torch
import numpy as np
from torch_geometric.data import Data

def generate_random_rotation_matrices(batch_size, device):
    """Generate batch of random 3D rotation matrices (batch_size, 3, 3)."""
    angles = torch.rand(batch_size, 3, device=device) * 2 * np.pi  # Random angles for X, Y, Z

    cx, cy, cz = torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])
    sx, sy, sz = torch.sin(angles[:, 0]), torch.sin(angles[:, 1]), torch.sin(angles[:, 2])

    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=1)
    ], dim=2)

    Ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=1)
    ], dim=2)

    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=1)
    ], dim=2)

    R = Rz @ Ry @ Rx  # (batch_size, 3, 3)
    return R

def augment_point_clouds_batch(data_list, device):
    """Augment a batch of point clouds with independent random 3D rotations in parallel."""
    batch_size = len(data_list)
    max_points = max(data.x.size(0) for data in data_list)

    # Pad the point clouds to same size
    padded = torch.zeros((batch_size, max_points, 3), device=device)
    mask = torch.zeros((batch_size, max_points), dtype=torch.bool, device=device)
    for i, data in enumerate(data_list):
        n = data.x.size(0)
        padded[i, :n] = data.x.to(device)
        mask[i, :n] = 1

    # Generate random rotations (batch_size, 3, 3)
    R = generate_random_rotation_matrices(batch_size, device)

    # Apply: (B, N, 3) @ (B, 3, 3)^T => (B, N, 3)
    rotated = torch.bmm(padded, R.transpose(1, 2))

    # Update data with rotated points
    augmented_data_list = []
    for i, data in enumerate(data_list):
        n = mask[i].sum().item()
        data.x = rotated[i, :n]
        augmented_data_list.append(data)

    return augmented_data_list


