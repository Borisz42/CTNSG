import random
import json
import concurrent.futures
from tqdm import tqdm
import subprocess
import time
import requests
import torch
import sys

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

DAG_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "nodes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"}
            }
        }
    },
    "required": ["goal", "nodes", "edges"],
    "additionalProperties": False
}

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
    
    num_nodes = len(nodes)
    if num_nodes < 5 or num_nodes > 15:
        return False # Enforce 5-15 node limit strictly at verification
        
    # Referential integrity
    for edge in edges:
        if len(edge) != 2:
            return False
        src, dst = edge
        if src < 0 or src >= num_nodes or dst < 0 or dst >= num_nodes:
            return False
            
    # Cycle detection
    if has_cycle(edges, num_nodes):
        return False
        
    return True

# ---------------------------------------------------------------------------
# 3. Dynamic Prompting
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
"""

# ---------------------------------------------------------------------------
# 4. Worker Execution
# ---------------------------------------------------------------------------
def worker_task(task_id):
    # Alternating load balance based on task ID parity (Port 8000 vs 8001)
    port = 8000 if task_id % 2 == 0 else 8001
    url = f"http://localhost:{port}/v1/chat/completions"
    
    prompt = generate_dynamic_prompt()
    payload = {
        "model": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 1024,
        "guided_json": DAG_SCHEMA # Triggers outlines constrained decoding in vLLM
    }
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code != 200:
            return None
            
        raw_content = response.json()['choices'][0]['message']['content']
        parsed = json.loads(raw_content)
        
        if verify_dag(parsed):
            parsed['trajectory_id'] = f"kaggle_synth_{task_id}"
            return parsed
            
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 5. Engine Orchestration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Launching Server A on GPU 0 (Port 8000)...")
    server_a = subprocess.Popen(
        "VLLM_ATTENTION_BACKEND=TORCH_SDPA CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server "
        "--model cyankiwi/Qwen3.5-9B-AWQ-4bit --quantization awq "
        "--port 8000 --max-model-len 2048 --gpu-memory-utilization 0.80 --max-num-seqs 32 --enforce-eager",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print("Launching Server B on GPU 1 (Port 8001)...")
    server_b = subprocess.Popen(
        "VLLM_ATTENTION_BACKEND=TORCH_SDPA CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server "
        "--model cyankiwi/Qwen3.5-9B-AWQ-4bit --quantization awq "
        "--port 8001 --max-model-len 2048 --gpu-memory-utilization 0.80 --max-num-seqs 32 --enforce-eager",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Health Check
    urls = ["http://localhost:8000/v1/models", "http://localhost:8001/v1/models"]
    ready = [False, False]
    
    print("Waiting for both engines to initialize CUDA graphs (this takes 1-2 minutes)...")
    while not all(ready):
        for i, url in enumerate(urls):
            if not ready[i]:
                try:
                    res = requests.get(url, timeout=2)
                    if res.status_code == 200:
                        ready[i] = True
                        print(f"Server on Port {8000 + i} is ready!")
                except requests.exceptions.RequestException:
                    pass
        time.sleep(5)
    print("Both GPUs fully armed and ready. Starting generation...")

    SAMPLES = 5000
    MAX_WORKERS = 64
    dataset = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks in large batches to prevent holding 5000 futures in memory if unnecessary
        futures = {executor.submit(worker_task, i): i for i in range(SAMPLES * 2)} # Over-generate to account for failures
        
        with tqdm(total=SAMPLES, desc="Distilling via Dual T4") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res:
                    dataset.append(res)
                    pbar.update(1)
                
                if len(dataset) >= SAMPLES:
                    break # Stop when we hit the target

    torch.save(dataset, "arbor_graphs_kaggle.pt")
    print(f"Successfully serialized {len(dataset)} valid, highly diverse samples.")

    # Teardown
    print("Tearing down vLLM servers...")
    server_a.terminate()
    server_b.terminate()
    print("GPU instances successfully released.")
