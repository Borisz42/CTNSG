"""
Graph VQ-Transformer (GVT) Tokenizer Module

This package implements the macroplanner's discrete topological graph generator, 
compressing continuous semantic nodes into a finite state space via 
Residual Vector Quantization (RVQ) and Graph Attention Networks (GAT).
"""

from .model import GraphVQTransformer
from .quantizer import ResidualVectorQuantizer
from .encoder import GraphEncoder
from .ordering import get_rcm_ordering

__all__ = [
    "GraphVQTransformer",
    "ResidualVectorQuantizer",
    "GraphEncoder",
    "get_rcm_ordering"
]
