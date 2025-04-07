import torch.nn as nn 
import utils
"""
- Given a point cloud, the amount of nodes, the MLP architecture and the radius
	1) select the nodes with farthes point sampling
	2) Group nodes by using a ball query of radius r for every selected node
	3) for **each node in the cloud** run the MLP on its features
	4) for each selected node, gather the embedding of its neighborhood using a maxpool operation
	- Now you have a new set of points and a new set of features, return it  
"""

class SetAbstractionLayer(nn.Module):
    def __init__(self, n_points, mlp, radius):
        super().__init__()
        self.n_points = n_points
        self.mlp = mlp 
        self.radius = radius

    def forward(self, point_cloud):
        # farthest point sampling
        new_point_cloud_indices = utils.farthest_point_sample(point_cloud, self.n_points)
        #new_point_cloud = 
        # group them using ball query
        # use knn for now, just because we need GPUs to use ball queries 
        knn_indices = utils.knn(point_cloud, new_point_cloud_indices, k = 8)
        # run the MLP on each node in the cloud
        batch_size, n_points, dim_points = point_cloud.shape 
        input_point_cloud = point_cloud.view(batch_size * n_points, dim_points)           # Flatten to 2D
        output_point_cloud = self.mlp(input_point_cloud)                # Apply MLP
        output_point_cloud = output_point_cloud.view(batch_size, n_points, dim_points)
        # gather the embedding for every selected node using a max pool
        #new_embeddings = 
        # return the new point cloud and the new embeddings
        #return new_point_cloud, new_embeddings
    
if __name__ == "__main__":
    mlp = nn.Linear(3, 10)
    sal = SetAbstractionLayer(50, mlp, 0.5)
    point_cloud, _ = utils.load_point_clouds("../data/unaligned_brains/")
    print(point_cloud.shape)
    sal.forward(point_cloud)