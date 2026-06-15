import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, num_layers: int = 4, heads: int = 4):
        """
        Uses PyTorch Geometric's Graph Attention Networks (GAT) to encode the 
        continuous input features into the latent dimension.
        """
        super().__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
            
        # Final layer: collapse heads into a single representation
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
        Returns:
            node_embeddings: Latent representations of nodes [num_nodes, hidden_channels]
            graph_embeddings: Pooled representation of the entire graph [batch_size, hidden_channels]
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = torch.relu(h)
                
        # Optional: create a global graph embedding if batch is provided
        # This is extremely useful for graph-level contrastive losses
        graph_emb = None
        if batch is not None:
            graph_emb = global_mean_pool(h, batch)
            
        return h, graph_emb
