import os
import json
import random
import time
import argparse
import concurrent.futures
import requests
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. Extreme Diversity Configurations
# ---------------------------------------------------------------------------
DOMAINS = [
    # --- Artificial Intelligence & Data Pipelines ---
    "Retrieval-Augmented Generation (RAG) Document Processing",
    "Large Language Model Distributed Training & Quantization",
    "Multi-Agent Reinforcement Learning Coordination",
    "Synthetic Data Distillation and Validation",
    "Deepfake and Manipulated Media Detection",
    "Computer Vision 3D Scene Reconstruction",
    "Federated Learning across Edge Devices",
    
    # --- Bioinformatics & Healthcare ---
    "Automated Drug Discovery and Lead Optimization",
    "CRISPR Gene Editing and Off-Target Analysis",
    "Protein Folding and Ligand Binding Simulation",
    "Multi-Modal MRI and X-Ray Diagnostic Triage",
    "Genomic Sequence Alignment and Assembly",
    "Epidemic Spread Modeling and Containment Strategy",
    "Brain-Computer Interface (BCI) Signal Decoding",

    # --- Physics, Space & Earth Sciences ---
    "Global Ocean and Climate Forecasting",
    "Low-Earth Orbit Satellite Network Routing",
    "Spacecraft Orbital Debris Collision Avoidance",
    "Computational Fluid Dynamics (CFD) Emulation",
    "Nuclear Fusion Plasma Boundary Optimization",
    "Wildfire Smoke Detection and Emergency Response",
    "Seismic Wavefield Modeling and Tomography",

    # --- Hardware & Engineering ---
    "Semiconductor Lithography and Yield Simulation",
    "VLSI Logic Synthesis and Hardware Routing",
    "Quantum State Preparation and Error Correction",
    "Computer-Aided Design (CAD) Parametric Modeling",
    "Neuromorphic Spiking Neural Network Compilation",
    "Robotic Assembly Line Material Handling",
    
    # --- Autonomous Systems & Robotics ---
    "Multi-Robot Warehouse Fulfillment Logistics",
    "Autonomous Vehicle Sensor Fusion & Motion Planning",
    "Drone-Based Search and Rescue Operations",
    "Humanoid Robot Dexterous Manipulation Control",
    "Spatio-Temporal Traffic Signal Orchestration",

    # --- Cybersecurity & IT Infrastructure ---
    "Zero-Day Malware Reverse Engineering",
    "Distributed Denial of Service (DDoS) Mitigation",
    "Enterprise Cloud CI/CD Deployment Pipeline",
    "Smart Contract Vulnerability Auditing",
    "6G Telecommunications Network Slicing",
    "Data Center Liquid Cooling and Power Optimization",

    # --- Finance, Legal & Economics ---
    "Algorithmic High-Frequency Arbitrage Trading",
    "Cross-Border Financial Fraud Investigation",
    "Dynamic Pricing and Yield Management",
    "Automated Legal Contract Review and Compliance",
    "Real-Time Bidding and Ad Attribution",

    # --- Media, Gaming & Graphics ---
    "Real-Time Multiplayer Game State Synchronization",
    "Procedural 3D Asset and Texture Generation",
    "Audio-Visual Speech Recognition and Translation",
    "Video Rendering and Ray-Tracing Pipeline",

    # --- Heavy Industry & Supply Chain ---
    "End-to-End E-Commerce Supply Chain Tracking",
    "Agricultural Crop Yield and Irrigation Management",
    "Predictive Maintenance for Manufacturing Equipment",
    "Water Treatment and Desalination Plant Control"
]

CONSTRAINTS = [
    "The workflow must contain deep causal dependencies where at least one critical path is 4 to 5 steps deep (e.g., A -> B -> C -> D), requiring strict sequential execution before parallel branches can merge.",
    "Ensure the graph has exactly 7 nodes and at least one bottleneck node where 3 separate tasks merge.",
    "Create a highly parallelized graph where 4 distinct processing pipelines occur simultaneously before culminating in a final aggregation step.",
    "Generate a deep, linear sequence with no parallel branches.",
    "Include a diamond-shaped dependency layout where a node splits into two parallel tasks, which then merge back into a single node.",
    "Create a structure with 2 distinct entry point nodes that operate completely independently before finally syncing at the final node."
]

# ---------------------------------------------------------------------------
# 2. Symbolic Verification (DFS Cycle Detection & Bounds)
# ---------------------------------------------------------------------------
def has_cycle(edges_list, num_nodes):
    from collections import defaultdict
    adj = defaultdict(list)
    for src, dst in edges_list:
        adj[src].append(dst)
        
    visited = [0] * num_nodes
    
    def dfs(node):
        if visited[node] == 1:
            return True
        if visited[node] == 2:
            return False
        
        visited[node] = 1
        for neighbor in adj[node]:
            if dfs(neighbor):
                return True
        visited[node] = 2
        return False

    for i in range(num_nodes):
        if visited[i] == 0:
            if dfs(i):
                return True
    return False

def verify_dag(parsed_json):
    nodes = parsed_json.get('nodes', [])
    edges = parsed_json.get('edges', [])
    goal = parsed_json.get('goal', '')
    
    if not nodes or not edges or not goal:
        return False

    num_nodes = len(nodes)
    if num_nodes < 5 or num_nodes > 15:
        return False # Enforce 5-15 node limit strictly at verification
        
    # Referential integrity
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            return False
        src, dst = edge
        if not isinstance(src, int) or not isinstance(dst, int):
            return False
        if src < 0 or src >= num_nodes or dst < 0 or dst >= num_nodes:
            return False
            
    # Cycle detection
    if has_cycle(edges, num_nodes):
        return False
        
    return True

# ---------------------------------------------------------------------------
# 3. Dynamic Prompting (Strict Schema instructions for LM Studio)
# ---------------------------------------------------------------------------
def generate_dynamic_prompt():
    domain = random.choice(DOMAINS)
    constraint = random.choice(CONSTRAINTS)
    return f"""You are an expert AI system architect. Your task is to generate a complex Directed Acyclic Graph (DAG) representing a logical workflow.

DOMAIN: {domain}
STRUCTURAL CONSTRAINT: {constraint}
NODE LIMIT: You must generate between 5 and 15 nodes.

The goal should be a realistic, 1-sentence user request for the {domain} industry.
Every node should represent an action_in_snake_case.
Edges must be represented as pairs of indices [source, destination], zero-indexed, pointing away from the root.
Strictly ensure there are NO cycles.

YOU MUST RESPOND ONLY WITH VALID JSON. Do not include markdown formatting or explanations. Use the exact structure below:
{{
    "goal": "A realistic 1-sentence user request here.",
    "nodes": ["action_one", "action_two", "action_three"],
    "edges": [[0, 1], [1, 2]]
}}
"""

# ---------------------------------------------------------------------------
# 4. Worker Execution (Targeting LM Studio)
# ---------------------------------------------------------------------------
def worker_task(url, task_id):
    prompt = generate_dynamic_prompt()
    payload = {
        "model": "local-model", # LM Studio usually ignores this and uses the loaded model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"HTTP {response.status_code}: {response.text}")
            return None
            
        message = response.json()['choices'][0]['message']
        raw_content = message.get('content', '')
        if not raw_content and 'reasoning_content' in message:
            raw_content = message.get('reasoning_content', '')
        
        # Clean up potential markdown formatting that local models sometimes leak
        raw_content = raw_content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        
        parsed = json.loads(raw_content)
        
        if verify_dag(parsed):
            # Map back to original dataset format expected by the pipeline
            return {
                "source": "circuitsynth_lm_studio",
                "goal": parsed['goal'],
                "node_names": parsed['nodes'],
                "text_edges": parsed['edges'],
                "trajectory_id": f"local_synth_{task_id}"
            }
        else:
            print(f"Validation failed for parsed JSON: {parsed}")
            return None
    except Exception as e:
        print(f"Exception during generation/parsing: {e}")
        return None

def generate_circuitsynth_dags(num_samples=5000, max_workers=4):
    url = "http://localhost:1234/v1/chat/completions"
    dataset = []
    
    print(f"Connecting to LM Studio at {url}...")
    print(f"Target Samples: {num_samples} | Max Workers: {max_workers}")
    
    # Test connection
    try:
        requests.get("http://localhost:1234/v1/models", timeout=5)
        print("Successfully connected to LM Studio API!")
    except requests.exceptions.RequestException:
        print("ERROR: Could not connect to LM Studio! Ensure the server is running on localhost:1234.")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        total_submitted = 0
        
        with tqdm(total=num_samples, desc="Distilling via LM Studio") as pbar:
            while len(dataset) < num_samples:
                # Fill the queue just enough to keep workers busy, without over-submitting
                while len(futures) < max_workers * 2 and total_submitted < num_samples * 5:
                    futures.add(executor.submit(worker_task, url, total_submitted))
                    total_submitted += 1
                
                if not futures:
                    break
                    
                # Wait for at least one to finish
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    res = fut.result()
                    if res and len(dataset) < num_samples:
                        dataset.append(res)
                        pbar.update(1)
            
            # Cancel any remaining futures so the script exits immediately
            for fut in futures:
                fut.cancel()

    print(f"CircuitSynth successfully distilled {len(dataset)} Silver-Standard DAGs.")
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CircuitSynth Distillation via LM Studio")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    args = parser.parse_args()

    os.makedirs("processed_data", exist_ok=True)
    dags = generate_circuitsynth_dags(num_samples=args.samples, max_workers=args.workers)
    
    if dags:
        output_path = 'processed_data/arbor_graphs_full.pt'
        torch.save(dags, output_path)
        print(f"Saved {len(dags)} DAGs to {output_path}")
