import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts.graph_schema import DiscourseGraph
from realizer.grammar.greatgramma import GreatGramma
from realizer.safety.safellm import SafeLLMExtractor, OptimalTransportMonitor
from realizer.parallel.mtp import DecompositionAndFill
# Assumes vnpool exists and has a projector
try:
    from realizer.vnpool.projector import VNProjector
except ImportError:
    class VNProjector:
        def __init__(self, *args): pass
        def forward(self, x): return x

class CTNSGRealizer:
    """
    High-Throughput Neuro-Symbolic Decoding Engine.
    Integrates VNPool, GREATGRAMMA, MTP, and SafeLLM.
    """
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 512):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Initialize sub-modules
        self.projector = VNProjector(hidden_dim, hidden_dim)
        self.grammar = GreatGramma(vocab_size, allowed_concepts=["System", "Macroplanner", "Graph"])
        self.mtp_engine = DecompositionAndFill(hidden_dim, vocab_size)
        self.safe_llm = SafeLLMExtractor()
        self.ot_monitor = OptimalTransportMonitor(threshold=0.3)
        
    def generate(self, graph: DiscourseGraph, schema: dict, context_lines: int) -> str:
        """
        Executes the fully guarded decoding process.
        """
        # 1. VNPool projection (mock continuous features)
        graph_embeddings = torch.randn(1, len(graph.nodes), self.hidden_dim)
        projected_prompts = self.projector.forward(graph_embeddings)
        
        # 2. Setup Grammar mask
        psc_mask = self.grammar.compile_schema(schema)
        
        # 3. MTP Parallel Loop (Simulated base model forward)
        def mock_base_model(input_ids):
            batch, seq = input_ids.shape
            logits = torch.randn(batch, seq, self.vocab_size)
            # Apply grammar mask to logits
            logits = self.grammar.apply_transducer_masking(logits, state_id=0, psc=psc_mask)
            
            hidden = torch.randn(batch, seq, self.hidden_dim)
            return logits, hidden, None
            
        initial_ids = torch.tensor([[1]])
        generated_ids = self.mtp_engine.generate_parallel(mock_base_model, initial_ids, max_new_tokens=20)
        
        # 4. Safety & Monitor checks
        mock_attention = torch.softmax(torch.randn(1, 8, generated_ids.size(1), 100), dim=-1)
        if self.ot_monitor.check_disengagement(mock_attention):
            print("[Warning] Optimal Transport monitor detected contextual disengagement!")
            
        mock_text = "The [Line 1] triggers [Line 2]."
        valid_lines = self.safe_llm.extract_and_verify(mock_text, context_lines)
        
        return {
            "text": mock_text,
            "valid_citations": valid_lines,
            "tokens_generated": generated_ids.size(1)
        }

if __name__ == "__main__":
    print("Testing Realizer Integration...")
    from contracts.graph_schema import SemanticNode, DiscourseGraph
    
    stub_graph = DiscourseGraph(
        graph_id="g1", 
        nodes=[SemanticNode(node_id="n1", concept="test", vq_index=1)], 
        edges=[]
    )
    
    realizer = CTNSGRealizer()
    schema = {"type": "object"}
    result = realizer.generate(stub_graph, schema, context_lines=5)
    print("Realizer Result:", result)
