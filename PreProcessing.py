import pandas as pd
import torch
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.spatial import KDTree
import numpy as np
import re
import csv 


DIST_MATRIX = "./data/phylo_trees/allspeciesList_distmat.txt"
POINT_CLOUDS_DIR = "./data/unaligned_brains"
PROCESSED_DATA_PATH = "./data/aligned_brains_point_clouds" 
DATA_DIR = "./data"

class PointCloud():
    def __init__(self, point_cloud_tensor, label):
        self.point_cloud_tensor = point_cloud_tensor
        self.label = label

def pack_clouds(point_clouds, labels):
    packed_point_clouds = []
    for pc, label in zip(point_clouds, labels):
        if label != -1:
            packed_point_clouds.append(PointCloud(pc, label))
    return packed_point_clouds

def load_point_clouds(directory):
    """
    Loads all point clouds from a directory, converts them into tensors, and
    stacks them into a single tensor.

    Parameters:
    - directory: str, path to the directory containing point cloud files

    Returns:
    - A list of N tensors with shape (M, 3), where N is the number of point clouds and
      M is the number of points in each point cloud.
    - A list of animals names
    """
    data = []
    animal_labels = []
    prev_name = ""
    cur_name = ""
    for filepath in glob.glob(os.path.join(directory, '*.txt')):
        # Load point cloud (3 floats per line, space-separated)
        points = []
        #print(f"filepath {filepath}")
        with open(filepath, 'r') as f:
            match = re.match(r"([A-Za-z]+)", os.path.basename(f.name))
            cur_name = match.group(1)
            #print(f"cur name = {cur_name}")
            if cur_name != prev_name:
                prev_name = cur_name 
            animal_labels.append(cur_name.lower())
            for line in f:
                if line.strip():  # skip empty lines
                    x, y, z = map(float, line.strip().split())
                    points.append([x, y, z])

        # Convert to tensor
        point_tensor = torch.tensor(points, dtype=torch.float32)
        data.append(point_tensor)
    return data, animal_labels


def visualize_point_cloud(point_cloud):
    """
    Visualizes a single 3D point cloud using Matplotlib.
    
    Parameters:
    - point_cloud: Tensor of shape (M, 3) where M is the number of points, and each point has x, y, z coordinates.
    """
    # Convert tensor to numpy array for plotting
    points = point_cloud.numpy()
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points in 3D space
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

def center_points_cloud(pointclouds):
    centroids = torch.mean(pointclouds, dim=1, keepdim=True)
    return pointclouds - centroids

# compute volume of a point_tensor via convec_hull approximation
def compute_convex_hull_volume(points_tensor):
    points = points_tensor.numpy()  # convert to numpy if it's a tensor
    hull = ConvexHull(points)
    return hull.volume

def compute_convex_hull_volume_tensor(point_clouds):
    """
    Computes the volume of each point cloud using the convex hull.
    
    Parameters:
    point_clouds (numpy.ndarray): Tensor of shape (N, M, 3) representing N point clouds, each with M points in 3D.
    
    Returns:
    numpy.ndarray: Array of shape (N,) containing the volume of each point cloud.
    """
    volumes = []
    
    for i in range(len(point_clouds)):
        points = point_clouds[i]
        
        # Compute convex hull only if there are enough points
        if points.shape[0] >= 4:  # At least 4 non-coplanar points are needed for a 3D volume
            hull = ConvexHull(points)
            volumes.append(hull.volume)
        else:
            volumes.append(0.0)  # If fewer than 4 points, volume is 0
    
    return np.array(volumes)

def euclidean_distance(point1, point2):
    """
    Computes the Euclidean distance between two points.
    
    Parameters:
    - point1: A tensor of shape (3,) representing a 3D point.
    - point2: A tensor of shape (3,) representing a 3D point.
    
    Returns:
    - The Euclidean distance between the two points.
    """
    return torch.sqrt(torch.sum((point1 - point2) ** 2))

def point_cloud_to_knn_graph(point_cloud, label, k=5):
    """
    Converts a point cloud into a k-NN graph suitable for PyTorch Geometric.
    
    Args:
        point_cloud (Tensor): A tensor of shape [num_points, 3] representing the 3D points.
        k (int): Number of nearest neighbors.
        
    Returns:
        Data: A PyTorch Geometric Data object.
    """
    num_points = point_cloud.size(0)
    
    # Compute node features (distance from the origin)
    distances_from_origin = torch.norm(point_cloud, dim=1, keepdim=True)  # Shape [num_points, 1]

    # Build k-NN graph using KDTree (CPU)
    kdtree = KDTree(point_cloud.numpy())
    edges = []
    edge_attrs = []

    for idx, point in enumerate(point_cloud):
        # Query the k+1 nearest (including itself)
        dists, neighbors = kdtree.query(point.numpy(), k=k+1)
        
        # Skip self-loop (neighbor[0] == idx)
        for neighbor_idx, dist in zip(neighbors[1:], dists[1:]):
            edges.append([idx, neighbor_idx])
            edge_attrs.append([dist])

    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)            # Shape [num_edges, 1]
    
    # Package into PyTorch Geometric Data object
    data = Data(x=distances_from_origin, pos=point_cloud, edge_index=edge_index, edge_attr=edge_attr, label=label)
    return data

def raw_point_cloud_to_data(point_cloud, label):
    data = Data(x = point_cloud, edge_index = torch.empty(2,0, dtype=torch.long), label = label)
    return data

def save_graph(data, file_path):
    """
    Saves a PyTorch Geometric graph data object to a file.
    
    Parameters:
    - data: The PyTorch Geometric Data object to save.
    - file_path: The file path to save the data.
    """

    torch.save(data, file_path)

def load_data(directory):
    data_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):  # Check if it's a file
            data = torch.load(filepath, weights_only=False)
            data_list.append(data)
    return data_list

def load_common_to_species(csv_file):
    """
    Loads a CSV file that maps common names (file names) to species names.
    """
    common_to_species = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if it exists
        for row in reader:
            common_name = row[0].lower()  # Assuming the common name is in the first column
            species_name = row[1].lower()  # Assuming species name is in the second column
            common_to_species[common_name] = species_name
    return common_to_species

def get_species_index(common_name, common_to_species, species_to_idx):
    """
    Given a common name, a map from common name to species name, and a dictionary mapping species names to indices,
    this function returns the index of the species corresponding to the common name.
    """
    # 1. Get the species name from the common name
    species_name = common_to_species.get(common_name)
    if not species_name:
        raise ValueError(f"Common name '{common_name}' not found in the map.")

    # 2. Get the index from the species-to-index dictionary
    species_index = species_to_idx.get(species_name)
    if species_index is None:
        raise ValueError(f"Species name '{species_name}' not found in the species-to-index dictionary.")

    return species_index

def load_distance_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Leggi header (nomi attributi)
    header = lines[0].strip().split()
    n = int(header[0])
    header = [w.lower() for w in header[1:]]

    species_to_index = dict(zip(header,list(range(n))))
    # Prepara una matrice numpy
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    # Riempie la matrice ignorando le etichette
    for i, line in enumerate(lines[1:]):
        parts = line.strip().split()
        row_values = parts[1:]  # Salta il nome della riga
        distance_matrix[i] = list(map(float, row_values))

    # Converti in tensor
    distance_tensor = torch.tensor(distance_matrix, dtype=torch.float32)

    return distance_tensor


if __name__ == "__main__":
    labels = []
    with open("./data/label_mapping.txt", 'r') as file:
        for line in file:
            # Convert each line to an integer and add to the list
            labels.append(int(line.strip()))
    data, animals = load_point_clouds(POINT_CLOUDS_DIR)
    print(f"Labels = {labels}")
    print(f"Animals = {animals}")
    iterator = reversed(list(enumerate(labels)))
    for i,label in iterator:
        if label == -1:
            labels.pop(i)
            data.pop(i)
        else:
            data[i] = center_points_cloud(data[i])
            bundle = raw_point_cloud_to_data(data[i], label)
            save_graph(bundle, os.path.join(PROCESSED_DATA_PATH, "brain"+str(i) + "_label" + str(label) ))