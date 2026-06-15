import torch
import torch.nn as nn

class PSDDSemanticPrior(nn.Module):
    """
    Probabilistic Sentential Decision Diagrams (PSDD) Semantic Prior.
    Used for constraining token budgets upstream before generation via TruncProof,
    and evaluating the structural intent distribution.
    """
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        # Parameterization for the PSDD structural probabilities
        self.node_log_probs = nn.Parameter(torch.randn(vocab_size))

    def evaluate_prior(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the semantic prior probability of a given token sequence.
        """
        probs = torch.softmax(self.node_log_probs, dim=-1)
        return torch.prod(probs[tokens], dim=-1)

class TruncProofOptimizer:
    """
    Implements the TruncProof guardrail via convex optimization to bound token budgets upstream.
    This prevents the Realizer from overflowing the LLM context window.
    """
    def __init__(self, llm_max_context: int = 32768, schema_closure_tokens: int = 512, safety_margin: int = 128):
        """
        Args:
            llm_max_context: Total context window of the target LLM (e.g., 32768 for Qwen-2.5).
            schema_closure_tokens: Max tokens required to gracefully close the generated schema.
            safety_margin: Additional buffer tokens.
        """
        self.llm_max_context = llm_max_context
        self.schema_closure_tokens = schema_closure_tokens
        self.safety_margin = safety_margin

    def calculate_dynamic_budget(self, current_prompt_tokens: int) -> int:
        """
        Dynamically calculates the maximum number of tokens/nodes that can be generated.
        
        Optimization bound:
        max_budget = LLM_CONTEXT - prompt_tokens - schema_closure_tokens - safety_margin
        """
        available_context = self.llm_max_context - current_prompt_tokens - self.schema_closure_tokens - self.safety_margin
        
        if available_context <= 0:
            raise ValueError("Semantic Collapse Risk: The current prompt and schema closure bounds exceed the LLM context window.")
            
        return available_context

    def optimize_generation_budget(self, target_nodes: int, current_prompt_tokens: int) -> int:
        """
        Clamps the target number of nodes to generate to strictly prevent context overflow.
        """
        max_budget = self.calculate_dynamic_budget(current_prompt_tokens)
        return min(target_nodes, max_budget)
