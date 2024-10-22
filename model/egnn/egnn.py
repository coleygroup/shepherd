import torch
import torch_scatter
from torch import Tensor
from typing import Callable, Dict, Optional, Tuple

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation=torch.nn.LeakyReLU(0.2), include_final_activation = True):
        super(MultiLayerPerceptron, self).__init__()
        
        first_layer_dim = hidden_dim if num_hidden_layers > 0 else output_dim
        self.mlp = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, first_layer_dim),
        ])
        for i in range(num_hidden_layers):
            self.mlp.append(activation)
            
            if i == num_hidden_layers - 1:
                self.mlp.append(torch.nn.Linear(hidden_dim, output_dim))
            else:
                self.mlp.append(torch.nn.Linear(hidden_dim, hidden_dim))
            
        if include_final_activation:
            self.mlp.append(activation)
        
    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x
                
            
class EGNN(torch.nn.Module):
    def __init__(self, node_embedding_dim, node_output_embedding_dim, edge_attr_dim, distance_expansion_dim = 32, normalize_distance_vectors = True, num_MLP_hidden_layers = 2, MLP_hidden_dim = 32):
        super(EGNN, self).__init__()
        
        self.distance_expansion = GaussianSmearing(0.0, 16.0, distance_expansion_dim)
        
        self.message_mlp_embedding_dim = node_embedding_dim
        self.message_mlp = MultiLayerPerceptron(
            node_embedding_dim + node_embedding_dim + distance_expansion_dim + edge_attr_dim,
            MLP_hidden_dim,
            self.message_mlp_embedding_dim,
            num_hidden_layers = num_MLP_hidden_layers,
            include_final_activation = True,
        )
        
        self.coordinate_mlp = torch.nn.Linear(self.message_mlp_embedding_dim, 1)
        
        self.node_mlp_embedding_dim = node_embedding_dim
        self.node_mlp = MultiLayerPerceptron(
            node_embedding_dim + self.message_mlp_embedding_dim,
            MLP_hidden_dim,
            self.node_mlp_embedding_dim,
            num_hidden_layers = num_MLP_hidden_layers,
            include_final_activation = True,
        )
        self.node_output_embedding_dim = node_output_embedding_dim
        self.node_output_embedding = torch.nn.Linear(self.node_mlp_embedding_dim, self.node_output_embedding_dim)
        
        self.normalize_distance_vectors = normalize_distance_vectors
    
    def forward(self, x, pos, edge_index, batch, edge_attr = None, pos_update_mask = None, residual_pos_update = False, residual_x_update = False):
        
        edge_distances = torch.linalg.norm(pos[edge_index[1]] - pos[edge_index[0]], axis = -1)
        if edge_attr is not None:
            messages = self.message_mlp(
                torch.concat([
                    x[edge_index[1]], x[edge_index[0]], self.distance_expansion(edge_distances), edge_attr,
                ], dim = -1)
            )
        else:
            messages = self.message_mlp(
                torch.concat([
                    x[edge_index[1]], x[edge_index[0]], self.distance_expansion(edge_distances),
                ], dim = -1)
            )
        messages_agg = torch_scatter.scatter_sum(
            messages, 
            edge_index[1], 
            dim_size = x.shape[0],
            dim = 0,
        ) # summing over incoming messages to target node
        
        
        edge_distance_vec = pos[edge_index[1]] - pos[edge_index[0]] # target - source 
        if self.normalize_distance_vectors:
            edge_distance_vec = edge_distance_vec / (torch.linalg.norm(edge_distance_vec, dim = -1, keepdim = True) + 1e-6)

        edge_weights = self.coordinate_mlp(messages) # essentially attention weights, without softmax normalization
        pos_update = torch_scatter.scatter_mean(
            edge_distance_vec * edge_weights, 
            edge_index[1], 
            dim_size = x.shape[0], 
            dim = 0,
        ) # scatter_mean normalizes by the number of edges to each node
        
        if pos_update_mask is not None:
            pos_update[pos_update_mask] = pos_update[pos_update_mask] * 0.0
        
        if residual_pos_update:
            pos_update = pos + pos_update
        
        
        x_update = self.node_mlp(
            torch.cat([x, messages_agg], dim = -1),
        )
        x_update = self.node_output_embedding(x_update)
        
        if residual_x_update:
            assert x.shape[-1] == self.node_output_embedding_dim
            x_update = x + x_update
        
        return x_update, pos_update
    
    