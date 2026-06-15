import torch
import torch.nn as nn

class SDRTMessagePassing(nn.Module):
    """
    Simulates a Graph Neural Network (GNN) message passing layer without PyTorch Geometric.
    Operates on dense adjacency matrices for SDRT graphs.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: Node features [batch_size, num_nodes, in_channels]
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        # Message passing: Aggregate neighbors
        # adj should ideally be symmetrically normalized, but for SDRT direct matmul works.
        msg = torch.bmm(adj, x)
        
        # Transform
        out = self.linear(msg)
        return self.activation(out)

class SDRTNodeClassifier(nn.Module):
    """
    SDRT-GNN Filtering Module.
    Takes an SDRT graph of utterances and classifies them as salient or non-salient.
    Prunes non-salient nodes to solve the "middle curse" in long-context parsing.
    """
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.gnn1 = SDRTMessagePassing(feature_dim, hidden_dim)
        self.gnn2 = SDRTMessagePassing(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Salience probability
        )

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Returns salience scores for each node.
        """
        x = self.gnn1(node_features, adj_matrix)
        x = self.gnn2(x, adj_matrix)
        
        scores = self.classifier(x)
        return scores.squeeze(-1)

    def prune(self, node_features: torch.Tensor, adj_matrix: torch.Tensor, threshold: float = 0.5):
        """
        Prunes non-salient utterances and returns the filtered features and adjacency matrix.
        (Batch size = 1 assumed for simplification in pruning topology)
        """
        scores = self.forward(node_features, adj_matrix)
        
        # Create a mask for salient nodes
        salient_mask = scores > threshold
        
        # In a real batch setup this requires padding handling, 
        # but for a single graph:
        if node_features.size(0) == 1:
            mask = salient_mask[0]
            pruned_features = node_features[0][mask].unsqueeze(0)
            
            # Prune adjacency matrix
            pruned_adj = adj_matrix[0][mask][:, mask].unsqueeze(0)
            
            return pruned_features, pruned_adj
            
        return node_features, adj_matrix # Batch mode requires specialized scatter/gather 
