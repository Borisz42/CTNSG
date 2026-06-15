import time
import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path so we can import from macroplanner
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from macroplanner.gvt.model import GraphVQTransformer
from orchestrator.reldit.model import RelDiT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

print("Loading curriculum graphs...")
curriculum_graphs = torch.load('processed_data/ctnsg_curriculum.pt')
print(f"Loaded {len(curriculum_graphs)} graphs.")

gvt = GraphVQTransformer(in_channels=256, hidden_channels=256, num_embeddings=64, num_quantizers=4).to(device)
reldit = RelDiT(vocab_size=64, d_model=256).to(device)

def benchmark():
    num_test_graphs = 200 # First graph is huge, rest are small. So 200 is a good sample.
    subset = curriculum_graphs[:num_test_graphs]
    
    print(f"Benchmarking forward pass on {num_test_graphs} graphs...")
    start_time = time.time()
    
    # We do a purely forward pass without gradients to simulate relative speed,
    # then multiply by ~3x to account for backward pass.
    with torch.no_grad():
        for graph in subset:
            nodes_full = graph['nodes'].to(device)
            max_seq = 4096
            for i in range(0, nodes_full.shape[0], max_seq):
                nodes = nodes_full[i : i + max_seq]
                edges = torch.empty((2,0), dtype=torch.long, device=device)
                
                # GVT pass
                out = gvt(nodes, edges)
                tokens = out['discrete_tokens'][:, 0].unsqueeze(0)
                
                # RelDiT pass
                _ = reldit(tokens)
            
    end_time = time.time()
    elapsed_forward = end_time - start_time
    
    # Multiply by 3 to estimate full training step (forward + backward + optimizer)
    elapsed_total_est = elapsed_forward * 3.0
    
    print(f"Forward pass time for {num_test_graphs} graphs: {elapsed_forward:.2f}s")
    
    # Graph 0 is massive, so simple averaging is misleading.
    # Total nodes in dataset vs nodes in subset
    total_nodes_subset = sum(g['nodes'].shape[0] for g in subset)
    total_nodes_all = sum(g['nodes'].shape[0] for g in curriculum_graphs)
    
    print(f"Subset nodes: {total_nodes_subset}, Total nodes: {total_nodes_all}")
    
    time_per_node = elapsed_total_est / total_nodes_subset
    
    total_epoch_time = time_per_node * total_nodes_all
    total_run_time = total_epoch_time * 100
    
    print(f"\n--- ESTIMATES ({device}) ---")
    print(f"Estimated time per Epoch: {total_epoch_time / 60:.2f} minutes")
    print(f"Estimated time for 100 Epochs: {total_run_time / 3600:.2f} hours")
    
benchmark()
