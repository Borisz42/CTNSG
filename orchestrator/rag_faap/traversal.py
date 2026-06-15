import torch
import torch.nn as nn
from typing import List, Tuple

class ConfidenceJudge(nn.Module):
    """
    Acts as a functional fallback for the LLM-as-a-judge.
    Evaluates the current state of retrieved context to determine if further traversal 
    (more graph hops) is required based on confidence thresholds.
    """
    def __init__(self, context_dim: int):
        super().__init__()
        self.judge = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, accumulated_context: torch.Tensor) -> torch.Tensor:
        # accumulated_context: [batch_size, context_dim]
        # returns confidence score [batch_size, 1]
        return self.judge(accumulated_context)


class DynamicConfidenceTraversal:
    """
    Replaces static hop limits in RAG graph traversal.
    Dynamically decides whether to fetch neighboring nodes based on semantic confidence.
    """
    def __init__(self, context_dim: int, confidence_threshold: float = 0.85, max_hops: int = 5):
        self.context_dim = context_dim
        self.confidence_threshold = confidence_threshold
        self.max_hops = max_hops
        self.judge = ConfidenceJudge(context_dim)
        
    def traverse(self, initial_query: torch.Tensor, graph_retriever_func) -> Tuple[torch.Tensor, int]:
        """
        Dynamically traverses the FAAP context graph.
        `graph_retriever_func` is a callable taking (current_nodes) and returning (new_nodes, new_context).
        """
        current_context = initial_query
        hops = 0
        
        while hops < self.max_hops:
            confidence = self.judge(current_context)
            
            # If confidence is high enough, we stop traversing
            if confidence.mean().item() >= self.confidence_threshold:
                break
                
            # Otherwise, fetch next hop
            current_context = graph_retriever_func(current_context)
            hops += 1
            
        return current_context, hops
