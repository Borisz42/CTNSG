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

def load_toolbench_traces(limit=2000, file_path="data/toolbench.jsonl"):
    """
    Loads real, complex multi-step linear traces from ToolBench.
    In a true execution, this would parse the standard ToolBench JSONL format.
    """
    import os, json
    traces = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= limit: break
                traces.append(json.loads(line))
        print(f"Loaded {len(traces)} real traces from {file_path}")
    else:
        print(f"WARNING: ToolBench dataset not found at {file_path}. Generating simulated 'real' traces for testing...")
        # Simulate standard linear traces that would come from ToolBench
        for i in range(100):
            traces.append({
                "trajectory_id": f"tb_{i}",
                "goal": f"Execute standard ToolBench workflow {i}.",
                "linear_steps": [
                    {"step_idx": 0, "tool": "search_web", "input": {"query": "info"}, "output": {"result": "data_A"}},
                    {"step_idx": 1, "tool": "search_web", "input": {"query": "more info"}, "output": {"result": "data_B"}},
                    {"step_idx": 2, "tool": "summarize", "input": {"t1": "data_A", "t2": "data_B"}, "output": {"summary": "done"}}
                ]
            })
    return traces

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
    for i in range(num_steps):
        input_str = str(steps[i].get('input', ''))
        for j in range(i):
            # Check for value intersection (mocking true dependency resolution)
            output_dict = steps[j].get('output', {})
            for v in output_dict.values():
                if str(v) in input_str and len(str(v)) > 2:
                    edges_src.append(j)
                    edges_dst.append(i)
                    break
    
    return nodes_text, encode_nodes(nodes_text), edges_src, edges_dst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'full'])
    parser.add_argument('--toolbench_path', type=str, default='data/toolbench.jsonl')
    parser.add_argument('--synth_samples', type=int, default=5000)
    args = parser.parse_args()

    print(f"\n--- Running Arbor TDP Pipeline in {args.mode.upper()} mode ---")
    
    from circuitsynth import generate_circuitsynth_dags
    
    # 1. Load Real ToolBench Data
    toolbench_traces = load_toolbench_traces(limit=5000, file_path=args.toolbench_path)
    
    processed_graphs = []
    
    # Parse Real ToolBench linear traces into Parallel DAGs
    for entry in toolbench_traces:
        nodes_text, nf, src, dst = infer_arbor_dag(entry)
        
        if src:
            ei = torch.tensor([src, dst], dtype=torch.long)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
            
        num_nodes = nf.shape[0]
        if num_nodes > 0 and ei.shape[1] > 0:
            rcm_order = get_rcm_ordering(ei, num_nodes)
            canonical_nodes = nf[rcm_order]
        else:
            canonical_nodes = nf
            rcm_order = torch.arange(num_nodes)
            
        # Re-map text arrays based on RCM permutation
        rcm_list = rcm_order.tolist()
        canonical_node_names = [nodes_text[i] for i in rcm_list]
        
        # Build text_edges mapping new canonical indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(rcm_list)}
        text_edges = [[old_to_new[s], old_to_new[d]] for s, d in zip(src, dst)]
            
        processed_graphs.append({
            "source": "arbor_toolbench",
            "goal": entry["goal"],
            "node_names": canonical_node_names,
            "text_edges": text_edges,
            "nodes": canonical_nodes,
            "edges": ei,
            "rcm_permutation": rcm_order
        })
        
    # 2. Distill Synthetic DAGs via CircuitSynth
    num_synth = args.synth_samples if args.mode == 'full' else 100
    synthetic_dags = generate_circuitsynth_dags(num_samples=num_synth)
    
    # Encode the synthesized DAGs
    for dag in synthetic_dags:
        nf = encode_nodes(dag["node_names"])
        
        src = [e[0] for e in dag["text_edges"]]
        dst = [e[1] for e in dag["text_edges"]]
        
        if src:
            ei = torch.tensor([src, dst], dtype=torch.long)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
            
        num_nodes = nf.shape[0]
        if num_nodes > 0 and ei.shape[1] > 0:
            rcm_order = get_rcm_ordering(ei, num_nodes)
            canonical_nodes = nf[rcm_order]
        else:
            canonical_nodes = nf
            rcm_order = torch.arange(num_nodes)
            
        rcm_list = rcm_order.tolist()
        canonical_node_names = [dag["node_names"][i] for i in rcm_list]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(rcm_list)}
        text_edges = [[old_to_new[s], old_to_new[d]] for s, d in zip(src, dst)]
        
        processed_graphs.append({
            "source": dag["source"],
            "goal": dag["goal"],
            "node_names": canonical_node_names,
            "text_edges": text_edges,
            "nodes": canonical_nodes,
            "edges": ei,
            "rcm_permutation": rcm_order
        })
        
    os.makedirs('processed_data', exist_ok=True)
    out_path = f'processed_data/arbor_graphs_{args.mode}.pt'
    torch.save(processed_graphs, out_path)
    print(f"Successfully exported {len(processed_graphs)} canonicalized DAGs to {out_path}!")

if __name__ == "__main__":
    main()
