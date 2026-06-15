import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath('.'))

from macroplanner.gvt.model import GraphVQTransformer
from macroplanner.reldit.model import RelDiT
from orchestrator.arbor.planner import ArborPlanner
from realizer.realizer import CTNSGRealizer
from contracts.graph_schema import DiscourseGraph, SemanticNode, SemanticEdge

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    print("\n--- 1. Initialize Global Orchestrator ---")
    planner = ArborPlanner(input_dim=512, hidden_dim=256).to(device)
    global_intent = torch.randn(1, 512).to(device)
    decoupled_plan = planner.decouple_plan(global_intent)
    print("Decoupled Plan Confidence:", decoupled_plan['confidence'])
    
    print("\n--- 2. GVT Training Loop ---")
    gvt = GraphVQTransformer(in_channels=256, hidden_channels=256, num_embeddings=64, num_quantizers=4).to(device)
    gvt_optimizer = optim.AdamW(gvt.parameters(), lr=3e-4)
    num_nodes = 5
    mock_node_features = torch.randn(num_nodes, 256).to(device)
    mock_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long).to(device)
    
    for epoch in range(1):
        gvt_optimizer.zero_grad()
        out = gvt(mock_node_features, mock_edge_index)
        quantized_latents = out['z_q']
        vq_loss = out['commit_loss']
        discrete_indices = out['discrete_tokens']
        
        recon_loss = nn.MSELoss()(quantized_latents, mock_node_features)
        total_loss = recon_loss + vq_loss
        total_loss.backward()
        gvt_optimizer.step()
        print(f"Epoch {epoch+1} | GVT Loss: {total_loss.item():.4f}")
        
    print("\n--- 3. RelDiT Training Loop ---")
    reldit = RelDiT(vocab_size=64, d_model=256).to(device)
    reldit_optimizer = optim.AdamW(reldit.parameters(), lr=1e-4)
    
    for epoch in range(1):
        reldit_optimizer.zero_grad()
        # discrete_indices shape from GVT is [num_nodes, N]. RelDiT expects [batch_size, seq_len]
        # We'll just take the first codebook index for mock and add batch dim
        tokens = discrete_indices[:, 0].unsqueeze(0)
        loss = reldit(tokens)
        loss.backward()
        reldit_optimizer.step()
        print(f"Epoch {epoch+1} | RelDiT Loss: {loss.item():.4f}")
        
    print("\n--- 4. Realizer Inference Pipeline ---")
    realizer = CTNSGRealizer(vocab_size=3200, hidden_dim=256)
    inference_graph = DiscourseGraph(
        graph_id="infer_001",
        nodes=[
            SemanticNode(node_id="n1", concept="System", vq_index=12),
            SemanticNode(node_id="n2", concept="Macroplanner", vq_index=45)
        ],
        edges=[
            SemanticEdge(source_id="n1", target_id="n2", relation_type="triggers")
        ]
    )
    schema = {"type": "object", "properties": {"output": {"type": "string"}}}
    result = realizer.generate(inference_graph, schema, context_lines=5)
    print("\n=== Realizer Final Output ===")
    print(result)
    
if __name__ == "__main__":
    run_test()
