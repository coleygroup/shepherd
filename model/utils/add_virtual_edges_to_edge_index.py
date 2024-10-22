import torch
from torch_geometric.nn import radius_graph
import numpy as np

def add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, batch):
    """
    Adds edges to edge_index that connect all (real) nodes to the virtual node(s)
    
    Arguments:
        edge_index -- torch.LongTensor with shape (2, N_edges)
        virtual_node_mask -- torch.BoolTensor with shape (N_nodes,)
        batch -- torch.LongTensor with shape (N_nodes,)
        
    Returns:
        new_edge_index -- updated edge_index with additional virtual edges
    """
    # edge_index (2, N_edges)
    # virtual_node_mask (N_nodes,) -- boolean tensor where True indicates a virtual node
    # batch (N_nodes,)
    
    # remove existing edges to/from virtual nodes, to avoid duplicating edges
    edge_mask = virtual_node_mask[edge_index[1]] | virtual_node_mask[edge_index[0]]
    edge_index_without_VN = edge_index[:, ~edge_mask]
    
    # create edge_index_VN that has edges between all real nodes and each VN
    edge_index_fully_connected = radius_graph(
        torch.zeros((virtual_node_mask.shape[0],3), device = batch.device),
        r = np.inf,
        batch = batch,
        max_num_neighbors = 1000000,
    ) # this excludes self-loops, by default
    edge_mask_fully_connected = virtual_node_mask[edge_index_fully_connected[1]] | virtual_node_mask[edge_index_fully_connected[0]]
    edge_index_VN = edge_index_fully_connected[:, edge_mask_fully_connected]
    
    # combine and return  
    new_edge_index = torch.cat([edge_index_without_VN, edge_index_VN], dim = 1)
    return new_edge_index