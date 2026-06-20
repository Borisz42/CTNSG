import torch
import networkx as nx
from macroplanner.reldit.model import RelDiT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reldit = RelDiT(vocab_size=65, d_model=256).to(device)

# Load weights if available
import os
export_dir = os.path.join(os.path.abspath('..'), 'ctnsg_export')
# just generating random topology
reldit.eval()
with torch.no_grad():
    gen_tokens = reldit.generate(batch_size=1, seq_len=64, device=device, use_critic=True)
    
curr_tokens = gen_tokens[0].tolist()
print("Generated Tokens:", curr_tokens)

G = nx.DiGraph()
for j in range(len(curr_tokens) - 1):
    u, v = curr_tokens[j], curr_tokens[j+1]
    G.add_edge(u, v)

# Let's say our subtasks have these codebook indices
subtask_indices = [12, 45, 8]

edges = []
for i in range(len(subtask_indices)):
    for j in range(len(subtask_indices)):
        if i == j: continue
        u = subtask_indices[i]
        v = subtask_indices[j]
        # if path exists, add edge
        if u in G and v in G and nx.has_path(G, u, v):
            edges.append((i, j))

print("Inferred Edges:", edges)
