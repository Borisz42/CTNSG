import os
import json
import random
import time
import argparse
import concurrent.futures
from tqdm import tqdm

try:
    from google import genai
    from google.genai import errors
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
except ImportError:
    genai = None
    retry = lambda *args, **kwargs: lambda f: f

def is_acyclic(num_nodes, edges):
    """Checks if a directed graph has cycles using DFS."""
    adj = {i: [] for i in range(num_nodes)}
    for src, dst in edges:
        adj[src].append(dst)
        
    visited = [0] * num_nodes
    def dfs(node):
        if visited[node] == 1: return False # Cycle detected
        if visited[node] == 2: return True
        visited[node] = 1
        for neighbor in adj[node]:
            if not dfs(neighbor): return False
        visited[node] = 2
        return True
        
    for i in range(num_nodes):
        if visited[i] == 0:
            if not dfs(i): return False
    return True

if genai:
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60), 
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((errors.APIError,))
    )
    def call_gemini(client, model_name, prompt):
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.7}
        )
else:
    def call_gemini(client, model_name, prompt):
        return None

def generate_single_dag(client, models_to_try, schema_instructions, actions, entities, use_api):
    """Worker function to generate a single valid DAG."""
    if use_api:
        raw_text = None
        for model_name in models_to_try:
            try:
                response = call_gemini(client, model_name, schema_instructions)
                raw_text = response.text
                break
            except Exception as e:
                continue
        
        if not raw_text:
            return None # All models failed (likely rate limits)

        try:
            if raw_text.startswith("```json"):
                raw_text = raw_text.strip("```json").strip("```").strip()
                
            data = json.loads(raw_text)
            goal = data.get("goal")
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            
            if not goal or not nodes or not isinstance(edges, list):
                return None
                
            # Symbolic Filtering: Referential Integrity
            num_nodes = len(nodes)
            valid_edges = []
            integrity_passed = True
            for edge in edges:
                if not isinstance(edge, list) or len(edge) != 2:
                    integrity_passed = False
                    break
                src, dst = edge
                if not (isinstance(src, int) and isinstance(dst, int)):
                    integrity_passed = False
                    break
                if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
                    integrity_passed = False
                    break
                valid_edges.append([src, dst])
                
            if not integrity_passed:
                return None
                
            # Symbolic Filtering: Acyclicity
            if not is_acyclic(num_nodes, valid_edges):
                return None
                
            return {
                "source": "circuitsynth",
                "goal": goal,
                "node_names": nodes,
                "text_edges": valid_edges
            }
        except Exception as e:
            return None
    else:
        # Fallback dummy generator
        num_nodes = random.randint(3, 7)
        nodes = random.choices(actions, k=num_nodes)
        goal = f"Perform {nodes[-1]} after running " + " and ".join(nodes[:-1]) + f" on {random.choice(entities)}"
        edges = []
        for dst in range(1, num_nodes):
            num_deps = random.randint(1, min(2, dst))
            deps = random.sample(range(dst), num_deps)
            for src in deps:
                edges.append([src, dst])
                
        return {
            "source": "circuitsynth_dummy",
            "goal": goal,
            "node_names": nodes,
            "text_edges": edges
        }

def generate_circuitsynth_dags(num_samples=1000, teacher_api_key=None, max_workers=5):
    """
    Implements the CircuitSynth Teacher-LLM distillation protocol with concurrency.
    """
    print(f"Running CircuitSynth Teacher-LLM Distillation Protocol...")
    print(f"Target Samples: {num_samples} | Max Workers: {max_workers}")
    
    api_key = teacher_api_key or os.environ.get("GEMINI_API_KEY")
    use_api = False
    
    if api_key and genai:
        client = genai.Client(api_key=api_key)
        use_api = True
        print("Using google.genai SDK for Teacher-LLM generation.")
    else:
        client = None
        print("Warning: No GEMINI_API_KEY found or dependencies missing. Falling back to dummy generator.")
        
    actions = ["fetch_data", "process_image", "summarize_text", "send_email", "query_db", "train_model", "deploy_app", "calculate_metrics", "generate_report"]
    entities = ["sales_data", "user_profile", "monthly_logs", "weather_info", "stock_prices", "server_metrics"]
    
    schema_instructions = '''
Generate a single complex task DAG. The output must be valid JSON with the following structure:
{
  "goal": "Raw user prompt describing the task. E.g. 'Compare the weather in Tokyo and Paris, then email me the result.'",
  "nodes": ["list", "of", "action_names", "in", "snake_case"],
  "edges": [[0, 1], [1, 2]] // A list of directed edges [src_index, dst_index] referencing indices in the nodes array.
}
Ensure the graph has at least 3 nodes, is a valid Directed Acyclic Graph (DAG), and references are strictly within the nodes array bounds.
'''

    models_to_try = [
        'gemini-3-flash-preview', 
        'gemini-3.1-flash-lite', 
        'gemini-2.5-flash', 
        'gemini-2.5-flash-lite', 
        'gemini-2.5-pro',
        'gemma-4-31b-it',
        'gemma-4-26b-a4b-it'
    ]

    synthetic_dags = []
    
    pbar = tqdm(total=num_samples, desc="Distilling DAGs", unit="DAG")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        
        while len(synthetic_dags) < num_samples:
            # Fill the worker pool
            while len(futures) < max_workers and len(synthetic_dags) + len(futures) < num_samples * 2:
                futures.add(executor.submit(generate_single_dag, client, models_to_try, schema_instructions, actions, entities, use_api))
                
            if not futures:
                break
                
            # Wait for at least one task to complete
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            
            for fut in done:
                res = fut.result()
                if res and len(synthetic_dags) < num_samples:
                    res['trajectory_id'] = f"synth_circuitsynth_{len(synthetic_dags)}"
                    synthetic_dags.append(res)
                    pbar.update(1)

    pbar.close()
    print(f"CircuitSynth successfully distilled {len(synthetic_dags)} Silver-Standard DAGs.")
    return synthetic_dags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CircuitSynth Distillation")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers")
    args = parser.parse_args()

    import torch
    os.makedirs("processed_data", exist_ok=True)
    dags = generate_circuitsynth_dags(num_samples=args.samples, max_workers=args.workers)
    
    output_path = 'processed_data/arbor_graphs_full.pt'
    torch.save(dags, output_path)
    print(f"Saved {len(dags)} DAGs to {output_path}")
