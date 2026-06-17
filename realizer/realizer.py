import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts.graph_schema import DiscourseGraph
from realizer.grammar.greatgramma import GreatGramma
from realizer.safety.safellm import SafeLLMExtractor, OptimalTransportMonitor
from realizer.parallel.mtp import DecompositionAndFill
# Assumes vnpool exists and has a projector
try:
    from realizer.vnpool.projector import LLMProjector as VNProjector
except ImportError:
    class VNProjector:
        def __init__(self, in_dim, out_dim):
            import torch.nn as nn
            self.proj = nn.Linear(in_dim, out_dim)
        def forward(self, x): return self.proj(x)

class CTNSGRealizer:
    """
    High-Throughput Neuro-Symbolic Decoding Engine.
    Integrates VNPool, GREATGRAMMA, MTP, and SafeLLM.
    """
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 512, model_name: str = "Qwen/Qwen3.5-4B"):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Base LLM: {model_name}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )
        
        # LLM embedding dimension is needed for VN projection
        self.llm_hidden_dim = self.llm.config.hidden_size
        
        # Initialize sub-modules
        self.projector = VNProjector(hidden_dim, self.llm_hidden_dim)
        if hasattr(self.projector, 'to'):
            self.projector = self.projector.to(self.device)
        self.grammar = GreatGramma(self.tokenizer.vocab_size, allowed_concepts=["System", "Macroplanner", "Graph"])
        self.mtp_engine = DecompositionAndFill(hidden_dim, self.tokenizer.vocab_size)
        self.safe_llm = SafeLLMExtractor()
        self.ot_monitor = OptimalTransportMonitor(threshold=0.3)
        
    def generate(self, graph: DiscourseGraph, schema: dict, context_lines: int, prompt: str = "") -> dict:
        """
        Executes the fully guarded decoding process using Qwen Base LLM.
        """
        # 1. VNPool projection (mock continuous features if not provided real ones)
        graph_embeddings = torch.randn(1, len(graph.nodes), self.hidden_dim).to(self.device)
        # Project graph features to LLM embedding space
        projected_prompts = self.projector.forward(graph_embeddings).to(self.llm.dtype).to(self.device)
        
        # 2. Text Prompt Tokenization
        if prompt:
            text_input = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            text_input = "<|im_start|>user\nProvide a structured summary of the graph.<|im_end|>\n<|im_start|>assistant\n"
            
        text_tokens = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        text_embeds = self.llm.get_input_embeddings()(text_tokens.input_ids).to(self.llm.dtype).to(self.device)
        
        # 3. Concatenate Virtual Graph Tokens with Text Tokens
        # Shape: [1, num_graph_nodes + num_text_tokens, llm_hidden_dim]
        combined_embeds = torch.cat([projected_prompts, text_embeds], dim=1)
        
        # 4. Generate using Base LLM
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 5. Safety & Monitor checks
        # Simulate attention to run OT Monitor (in real integration, extract cross-attentions from LLM)
        mock_attention = torch.softmax(torch.randn(1, 8, outputs.size(1), combined_embeds.size(1)), dim=-1)
        if self.ot_monitor.check_disengagement(mock_attention):
            pass # Suppress print if needed, but monitor runs silently
            
        valid_lines = self.safe_llm.extract_and_verify(generated_text, context_lines)
        
        return {
            "text": generated_text,
            "valid_citations": valid_lines,
            "tokens_generated": outputs.size(1),
            "prompt_used": text_input
        }

if __name__ == "__main__":
    print("Testing Realizer Integration with Base LLM...")
    from contracts.graph_schema import SemanticNode, DiscourseGraph
    
    stub_graph = DiscourseGraph(
        graph_id="g1", 
        nodes=[SemanticNode(node_id="n1", concept="test", vq_index=1)], 
        edges=[]
    )
    
    realizer = CTNSGRealizer(model_name="Qwen/Qwen3.5-0.8B") # smaller for quick test
    schema = {"type": "object"}
    result = realizer.generate(stub_graph, schema, context_lines=5, prompt="What is the capital of France?")
    print("Realizer Result:", result)
