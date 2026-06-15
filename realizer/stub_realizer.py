import sys
import os
import json

# Ensure we can import from the contracts module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts.graph_schema import DiscourseGraph

try:
    import torch
    import outlines
    from pydantic import BaseModel
    import transformers
    HAS_DEPS = True
except ImportError:
    print("Dependencies not found. We will mock the output for validation.")
    HAS_DEPS = False
    class BaseModel: pass

class RealizedOutput(BaseModel):
    summary: str
    nodes_mentioned: list[str]
    relation_chain: str

def load_graph(filepath: str) -> DiscourseGraph:
    with open(filepath, "r") as f:
        return DiscourseGraph.from_json(f.read())

def run_stub_realizer(graph_path: str):
    print(f"Loading graph from {graph_path}...")
    graph = load_graph(graph_path)
    print(f"Graph loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Initializing {model_name}...")
    
    if not HAS_DEPS:
        print("\n[Mocking due to missing dependencies...]")
        print(json.dumps({
            "summary": "The System Initiation triggers the Macroplanner which generates the Discrete Graph.",
            "nodes_mentioned": ["System Initiation", "Macroplanner", "Discrete Graph"],
            "relation_chain": "n1 triggers n2 generates n3"
        }, indent=2))
        return
        
    try:
        # We use bitsandbytes 4-bit quantization if we're fitting this into the 8GB VRAM
        model = outlines.models.transformers(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"load_in_4bit": True} if torch.cuda.is_available() else {}
        )
        
        # Enforce the generation to strictly follow our RealizedOutput pydantic schema
        generator = outlines.generate.json(model, RealizedOutput)
        
        # Construct the prompt based on the graph
        node_descriptions = ", ".join([f"{n.node_id}: {n.concept}" for n in graph.nodes])
        edge_descriptions = ", ".join([f"{e.source_id} {e.relation_type} {e.target_id}" for e in graph.edges])
        
        prompt = f"""
        You are the Realizer module of the CTNSG framework.
        Convert the following discrete semantic graph into natural language.
        
        Nodes: {node_descriptions}
        Edges: {edge_descriptions}
        
        Provide your output strictly in the requested JSON format.
        """
        
        print("Generating O(1) grammar-constrained output...")
        result = generator(prompt)
        print("\n=== Generation Result ===")
        print(result.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"\n[Warning] Model execution failed or skipped. Ensure you have CUDA and adequate VRAM.")
        print(f"Error details: {e}")
        print("\nMocking the expected output for validation purposes...")
        print(json.dumps({
            "summary": "The System Initiation triggers the Macroplanner which generates the Discrete Graph.",
            "nodes_mentioned": ["System Initiation", "Macroplanner", "Discrete Graph"],
            "relation_chain": "n1 triggers n2 generates n3"
        }, indent=2))

if __name__ == "__main__":
    graph_path = os.path.join(os.path.dirname(__file__), "..", "macroplanner", "stub_graph.json")
    if not os.path.exists(graph_path):
        print(f"Graph file not found at {graph_path}. Please run macroplanner/stub_graph_generator.py first.")
        sys.exit(1)
        
    run_stub_realizer(graph_path)
