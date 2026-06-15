import os
import argparse
import sys
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from macroplanner.gvt.ordering import get_rcm_ordering

print("Initializing SentenceTransformer for Arbor nodes...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
projection = nn.Linear(384, 256)
with torch.no_grad():
    projection.weight.fill_(0.01)
    projection.bias.fill_(0.0)

def encode_nodes(texts):
    with torch.no_grad():
        embeddings = encoder.encode(texts, convert_to_tensor=True)
        projected = projection(embeddings.cpu())
    return projected

def get_mock_toolbench():
    """Simulates a complex linear tool-use trajectory."""
    return [
        {
            "trajectory_id": "tb_001",
            "goal": "Compare the weather in New York and London and email the result to Alice.",
            "linear_steps": [
                {"step_idx": 0, "tool": "get_weather", "input": {"city": "New York"}, "output": {"temp": "22C"}},
                {"step_idx": 1, "tool": "get_weather", "input": {"city": "London"}, "output": {"temp": "15C"}},
                {"step_idx": 2, "tool": "compare_weather", "input": {"temp1": "22C", "temp2": "15C"}, "output": {"diff": "New York is warmer"}},
                {"step_idx": 3, "tool": "send_email", "input": {"to": "Alice", "body": "New York is warmer"}, "output": {"status": "sent"}}
            ]
        }
    ]

def infer_arbor_dag(trajectory):
    """
    Parses a linear trace into a true parallel DAG based on input/output dependencies.
    """
    steps = trajectory['linear_steps']
    num_steps = len(steps)
    
    nodes_text = [s['tool'] for s in steps]
    edges_src = []
    edges_dst = []
    
    # Simple dependency inference: Check if a step's input contains substrings from a previous step's output
    # If not, they are independent and can be executed in parallel!
    for i in range(num_steps):
        input_str = str(steps[i]['input'])
        has_dependency = False
        for j in range(i):
            output_str = str(steps[j]['output'])
            # Check for value intersection (mocking true dependency resolution)
            for v in steps[j]['output'].values():
                if v in input_str:
                    edges_src.append(j)
                    edges_dst.append(i)
                    has_dependency = True
                    break
    
    return encode_nodes(nodes_text), edges_src, edges_dst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'full'])
    args = parser.parse_args()

    print(f"\n--- Running Arbor TDP Pipeline in {args.mode.upper()} mode ---")
    
    ds = get_mock_toolbench()
    processed_graphs = []
    
    for entry in ds:
        nf, src, dst = infer_arbor_dag(entry)
        
        if src:
            ei = torch.tensor([src, dst], dtype=torch.long)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
            
        # Apply RCM Canonicalization
        num_nodes = nf.shape[0]
        if num_nodes > 0 and ei.shape[1] > 0:
            rcm_order = get_rcm_ordering(ei, num_nodes)
            canonical_nodes = nf[rcm_order]
        else:
            canonical_nodes = nf
            rcm_order = torch.arange(num_nodes)
            
        processed_graphs.append({
            "source": "arbor_toolbench",
            "nodes": canonical_nodes,
            "edges": ei,
            "rcm_permutation": rcm_order
        })
        
        # Mathematical verification: Steps 0 and 1 should have NO incoming edges (Parallel execution)
        incoming_edges = dst
        parallel_nodes = [i for i in range(num_nodes) if i not in incoming_edges]
        print(f"Topology Assert: Independent Parallel Nodes = {parallel_nodes}")
        assert 0 in parallel_nodes and 1 in parallel_nodes, "DAG failed to parallelize independent steps!"
        
    os.makedirs('processed_data', exist_ok=True)
    out_path = f'processed_data/arbor_graphs_{args.mode}.pt'
    torch.save(processed_graphs, out_path)
    print(f"Successfully exported {len(processed_graphs)} canonicalized DAGs to {out_path}!")

if __name__ == "__main__":
    main()
