# Module 1: Graph Vector Tokenization (GVT)

This module forms the foundation of the Macroplanner in the CTNSG framework. Its primary responsibility is to encode continuous, high-dimensional topological graph data into discrete, compressible sequences of tokens, allowing topological generation to be treated as a standard discrete sequence modeling task.

## Key Components

### 1. `encoder.py` (Graph VQ-Transformer - GVT)
We utilize a pure **TransformerEncoder** architecture. The continuous vectors are processed through multi-head self-attention, allowing them to capture rich topological context before quantization. This removes the dependency on PyTorch Geometric's GAT layers, operating purely on canonicalized sequences.

### 2. `rope.py` (Rotary Position Embeddings - RoPE)
Implements 1D Rotary Position Embeddings optimized for sequences derived from graphs. When paired with RCM, it mathematically guarantees exact structural reconstruction without needing continuous edge attributes.

### 3. `ordering.py` (Reverse Cuthill-McKee Sequence Canonicalization)
Graphs lack a natural left-to-right order, making them notoriously difficult for sequence models to predict. This module implements the **Reverse Cuthill-McKee (RCM)** heuristic algorithm. By minimizing the bandwidth of the adjacency matrix, RCM provides a canonical, structurally meaningful sequence order, drastically reducing the state-space ambiguity for the downstream diffusion model.

### 3. `quantizer.py` (Residual Vector Quantization - RVQ)
Instead of flattening the graph into a single enormous codebook (which leads to representation collapse), we use **Residual Vector Quantization** with a Gumbel-Softmax reparameterization trick. 
- **Codebook Size ($K$):** 64
- **Depth ($N$):** 4
This means every node is quantized into an array of 4 discrete indices, preserving fine-grained hierarchical details while ensuring the codebooks remain small enough to train stably on 16GB Kaggle instances.

### 4. `model.py` (Master GVT Tokenizer)
The orchestrator script that takes raw `torch_geometric` graph batches, computes the RCM ordering, extracts the GAT continuous features, and compresses them through the RVQ layers. It outputs both the discrete indices for downstream sequence training and the continuous embeddings for reconstruction loss.
