import os
import argparse
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn

print("Initializing SentenceTransformer for Verification nodes...")
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

def get_mock_verification_data():
    """Simulates HaluEval hallucinated MAS interaction and ChatEval debate scoring."""
    return [
        {
            "type": "saiguard_contagion",
            "scenario": "Agent A hallucinates and passes it to B and C.",
            "agents": ["Agent_A_Generator", "Agent_B_Reviewer", "Agent_C_Executor"],
            "messages": [
                "The capital of France is Berlin.", # Hallucination (Node 0)
                "I see Berlin is the capital, I will log it.", # Contagion B (Node 1)
                "Executing search for Berlin tourism." # Contagion C (Node 2)
            ],
            "interaction_edges": [(0, 1), (0, 2)], # A -> B, A -> C
            "anomalous_edges": [(0, 1), (0, 2)] # SAIGuard targets
        },
        {
            "type": "brick_armor_debate",
            "scenario": "Three agents debate a math problem. Agent 3 is a semantic outlier.",
            "agents": ["Agent_1", "Agent_2", "Agent_3"],
            "messages": [
                "2+2=4 because of standard arithmetic.",
                "Yes, 4 is the correct integer sum.",
                "2+2=5 if we use Orwellian logic." # Semantic Outlier
            ],
            # 6 Capability Dimensions: [Math, Logic, Coding, Web, Extraction, Reasoning]
            "capability_scores": [
                [0.9, 0.9, 0.1, 0.1, 0.1, 0.8], # Agent 1
                [0.8, 0.8, 0.1, 0.1, 0.1, 0.7], # Agent 2
                [0.1, 0.2, 0.1, 0.1, 0.1, 0.9]  # Agent 3 (Low math, high abstract reasoning)
            ],
            "execution_costs": [0.01, 0.01, 0.05],
            "sod_outlier_mask": [0, 0, 1] # Agent 3 is the semantic outlier
        }
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'full'])
    args = parser.parse_args()

    print(f"\n--- Running Verification Pipeline (SAIGuard & Brick) in {args.mode.upper()} mode ---")
    
    ds = get_mock_verification_data()
    processed_data = []
    
    for entry in ds:
        nf = encode_nodes(entry['messages'])
        
        if entry['type'] == 'saiguard_contagion':
            src = [e[0] for e in entry['interaction_edges']]
            dst = [e[1] for e in entry['interaction_edges']]
            ei = torch.tensor([src, dst], dtype=torch.long)
            
            anomaly_mask = torch.ones(ei.shape[1], dtype=torch.float32) # Label 1 for anomalous contagion edges
            
            processed_data.append({
                "type": "saiguard",
                "nodes": nf,
                "edges": ei,
                "anomaly_mask": anomaly_mask
            })
            print(f"SAIGuard Contagion Topology: {nf.shape[0]} Agents, {ei.shape[1]} Contagion Edges")
            
        elif entry['type'] == 'brick_armor_debate':
            cap_scores = torch.tensor(entry['capability_scores'], dtype=torch.float32)
            costs = torch.tensor(entry['execution_costs'], dtype=torch.float32)
            sod_mask = torch.tensor(entry['sod_outlier_mask'], dtype=torch.long)
            
            processed_data.append({
                "type": "armor_mad",
                "nodes": nf,
                "capability_scores": cap_scores,
                "execution_costs": costs,
                "sod_mask": sod_mask
            })
            print(f"Brick/ARMOR Debate Topology: {nf.shape[0]} Agents, SOD Outliers Detected: {sod_mask.sum().item()}")
            
    os.makedirs('processed_data', exist_ok=True)
    out_path = f'processed_data/verification_graphs_{args.mode}.pt'
    torch.save(processed_data, out_path)
    print(f"Successfully exported {len(processed_data)} verification topologies to {out_path}!")

if __name__ == "__main__":
    main()
