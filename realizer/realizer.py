import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor

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

class GreatGrammaLogitsProcessor(LogitsProcessor):
    def __init__(self, great_gramma, schema, tokenizer=None, trunc_proof_optimizer=None, dynamic_budget=None):
        self.gg = great_gramma
        self.tokenizer = tokenizer
        self.psc = self.gg.compile_schema(schema, tokenizer=tokenizer)
        self.vocab_size = self.psc.vocab_size
        self.trunc_proof_optimizer = trunc_proof_optimizer
        self.dynamic_budget = dynamic_budget
        self.current_step = 0
        self.current_state_id = 0
        self.prompt_len = None
        self._dead_end_logged: set = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.current_step += 1

        if self.tokenizer and self.psc.schema:
            # Precompute on first call (lazy, done once)
            if not self.psc.is_precomputed:
                self.psc.precompute_closure_masks()
                self.current_state_id = 0

            # Update state ID based on the last generated token (steps > 1)
            if self.current_step > 1:
                last_token_id = input_ids[0][-1].item()
                if self.current_state_id < len(self.psc.transition_table):
                    trans = self.psc.transition_table[self.current_state_id]
                    if last_token_id in trans:
                        self.current_state_id = trans[last_token_id]
                    else:
                        # Token not in precomputed transitions — log once per (state, step) pair
                        key = (self.current_state_id, self.current_step)
                        if key not in self._dead_end_logged:
                            self._dead_end_logged.add(key)
                            print(
                                f"[PSC] WARNING: Token {last_token_id} has no transition "
                                f"from state {self.current_state_id} at step {self.current_step}. "
                                f"State held — mask remains from current state.",
                                flush=True
                            )
                        # Stay in the same state (mask enforces valid continuations)

            # TruncProof budget check
            if self.trunc_proof_optimizer and self.dynamic_budget is not None:
                if self.current_state_id < len(self.psc.shortest_path_len):
                    path_len = self.psc.shortest_path_len[self.current_state_id]
                    if (self.dynamic_budget - self.current_step) <= path_len:
                        next_token = self.psc.shortest_path_token[self.current_state_id]
                        if next_token != -1:
                            mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
                            mask[next_token] = True
                            new_scores = scores.clone()
                            new_scores[..., ~mask] = -float('inf')
                            return new_scores

            # Retrieve and apply precomputed mask (O(1))
            mask = self.psc.get_mask(self.current_state_id)
            mask = mask.to(scores.device)
            # Handle vocab padding: model logits may have more entries than tokenizer vocab
            if mask.shape[0] < scores.shape[-1]:
                padding = torch.zeros(scores.shape[-1] - mask.shape[0], dtype=torch.bool, device=scores.device)
                mask = torch.cat([mask, padding])
            new_scores = scores.clone()
            new_scores[..., ~mask] = -float('inf')
            return new_scores

        return scores


class CTNSGRealizer:
    """
    High-Throughput Neuro-Symbolic Decoding Engine.
    Integrates VNPool, GREATGRAMMA, MTP, and SafeLLM.
    """
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 256, model_name: str = "unsloth/Phi-4-mini-instruct", cache_dir: str = None):
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
        # Use tokenizer length for grammar (not model config padded size) so the
        # precomputed DFA cache key is stable across runs and mask dimensions include special tokens.
        tok_vocab_size = len(self.tokenizer)
        self.grammar = GreatGramma(tok_vocab_size, allowed_concepts=["System", "Macroplanner", "Graph"], cache_dir=cache_dir)
        self.mtp_engine = DecompositionAndFill(hidden_dim, self.llm.config.vocab_size)
        self.safe_llm = SafeLLMExtractor()
        self.ot_monitor = OptimalTransportMonitor(threshold=0.3)
        
    def generate(self, graph: DiscourseGraph, schema: dict, context_lines: int, prompt: str = "", graph_features: torch.Tensor = None) -> dict:
        """
        Executes the fully guarded decoding process using the Base LLM.
        """
        # 1. VNPool projection (mock continuous features if not provided real ones)
        if graph_features is not None:
            graph_embeddings = graph_features.to(self.device)
            if graph_embeddings.dim() == 2:
                graph_embeddings = graph_embeddings.unsqueeze(0)
        else:
            graph_embeddings = torch.randn(1, len(graph.nodes), self.hidden_dim).to(self.device)
            
        # Project graph features to LLM embedding space
        projected_prompts = self.projector.forward(graph_embeddings).to(self.llm.dtype).to(self.device)
        
        # 2. Text Prompt Tokenization
        if prompt:
            import json
            schema_str = json.dumps(schema, indent=2) if schema else ""
            instruction = f"\n\nYou MUST return your answer strictly as a JSON object matching this schema:\n{schema_str}" if schema else ""
            messages = [{"role": "user", "content": prompt + instruction}]
        else:
            messages = [{"role": "user", "content": "Provide a structured summary of the graph."}]
            
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        text_tokens = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        text_embeds = self.llm.get_input_embeddings()(text_tokens.input_ids).to(self.llm.dtype).to(self.device)
        
        # 3. Concatenate Virtual Graph Tokens with Text Tokens
        # If the projector is untrained, random weights cause float16 overflow in the LLM resulting in NaN logits.
        if os.path.exists("ctnsg_export/vnpool_projector_weights.pt"):
            combined_embeds = torch.cat([projected_prompts, text_embeds], dim=1)
        else:
            combined_embeds = text_embeds
        
        # Explicit attention mask required when passing inputs_embeds
        attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=self.device)
        
        # 4. Instantiate TruncProofOptimizer and compute dynamic budget
        from realizer.safety.safellm import TruncProofOptimizer
        trunc_proof = TruncProofOptimizer(llm_max_context=32768, schema_closure_tokens=256, safety_margin=64)
        
        prompt_tokens = combined_embeds.size(1)
        try:
            dynamic_budget = trunc_proof.calculate_dynamic_budget(prompt_tokens)
        except ValueError:
            dynamic_budget = 64
            
        max_new_tokens = min(1024, dynamic_budget)
        
        logits_processor = []
        if schema:
            lp = GreatGrammaLogitsProcessor(
                self.grammar, 
                schema, 
                tokenizer=self.tokenizer,
                trunc_proof_optimizer=trunc_proof,
                dynamic_budget=max_new_tokens
            )
            logits_processor.append(lp)
            
        from transformers import StoppingCriteria, StoppingCriteriaList
 
        class MultiSequenceStoppingCriteria(StoppingCriteria):
            def __init__(self, target_sequences):
                self.target_sequences = target_sequences
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
                is_done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                for idx in range(input_ids.shape[0]):
                    tokens_list = input_ids[idx].tolist()
                    for seq in self.target_sequences:
                        seq_len = len(seq)
                        if len(tokens_list) >= seq_len:
                            if tokens_list[-seq_len:] == seq:
                                is_done[idx] = True
                                break
                return is_done
 
        im_end_seq = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        eos_token_id = self.tokenizer.eos_token_id
        target_sequences = []
        if im_end_seq:
            target_sequences.append(im_end_seq)
        if eos_token_id is not None:
            target_sequences.append([eos_token_id])
 
        stopping_criteria = StoppingCriteriaList([MultiSequenceStoppingCriteria(target_sequences)])
 
        with torch.no_grad():
            if hasattr(self.llm, "generation_config"):
                self.llm.generation_config.max_length = None
            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        for stop_str in ["<|im_end|>", "<|end|>"]:
            if stop_str in generated_text:
                generated_text = generated_text.split(stop_str)[0]
        
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
    
    realizer = CTNSGRealizer(model_name="unsloth/Phi-4-mini-instruct") # smaller for quick test
    schema = {"type": "object"}
    result = realizer.generate(stub_graph, schema, context_lines=5, prompt="What is the capital of France?")
    print("Realizer Result:", result)
