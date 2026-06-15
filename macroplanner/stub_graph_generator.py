import sys
import os
import json

# Ensure we can import from the contracts module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contracts.graph_schema import SemanticNode, SemanticEdge, DiscourseGraph

def generate_stub_graph() -> DiscourseGraph:
    """Generates a hardcoded semantic discourse graph."""
    nodes = [
        SemanticNode(node_id="n1", concept="System Initiation", vq_index=12, attributes={"type": "event"}),
        SemanticNode(node_id="n2", concept="Macroplanner", vq_index=45, attributes={"type": "module"}),
        SemanticNode(node_id="n3", concept="Discrete Graph", vq_index=89, attributes={"type": "data_structure"}),
    ]
    
    edges = [
        SemanticEdge(source_id="n1", target_id="n2", relation_type="triggers"),
        SemanticEdge(source_id="n2", target_id="n3", relation_type="generates"),
    ]
    
    graph = DiscourseGraph(
        graph_id="g_stub_001",
        nodes=nodes,
        edges=edges,
        metadata={"description": "A stubbed causal graph for testing the Realizer."}
    )
    return graph

if __name__ == "__main__":
    graph = generate_stub_graph()
    output_path = os.path.join(os.path.dirname(__file__), "stub_graph.json")
    with open(output_path, "w") as f:
        f.write(graph.to_json())
    print(f"Successfully generated stub graph to {output_path}")
