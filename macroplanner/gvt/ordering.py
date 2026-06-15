import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

def get_rcm_ordering(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Computes the Reverse Cuthill-McKee (RCM) ordering for a given graph topology.
    
    This canonicalizes the graph node ordering to ensure topological symmetry 
    and invariance to isomorphic permutations, which is critical before 
    passing the graph into a Transformer.
    
    Args:
        edge_index (torch.Tensor): PyG edge_index of shape [2, num_edges]
        num_nodes (int): Total number of nodes in the graph.
        
    Returns:
        torch.Tensor: A 1D tensor of shape [num_nodes] containing the permuted indices.
    """
    # Convert edge_index to numpy for scipy
    edges = edge_index.cpu().numpy()
    
    # Create a symmetric adjacency matrix (RCM expects symmetric matrix for undirected connectivity)
    # We add 1 for weights, though RCM only cares about structure
    data = np.ones(edges.shape[1])
    
    # Make symmetric (undirected) to ensure RCM properly traverses connected components
    row = np.concatenate([edges[0], edges[1]])
    col = np.concatenate([edges[1], edges[0]])
    data_sym = np.concatenate([data, data])
    
    adj = csr_matrix((data_sym, (row, col)), shape=(num_nodes, num_nodes))
    
    # Compute RCM permutation
    perm = reverse_cuthill_mckee(adj, symmetric_mode=True)
    
    return torch.tensor(perm.copy(), dtype=torch.long, device=edge_index.device)
