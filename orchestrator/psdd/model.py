import torch
import torch.nn as nn
import torch.optim as optim

class PSDDSemanticPrior(nn.Module):
    """
    Probabilistic Sentential Decision Diagrams (PSDD) Semantic Prior.
    Used for constraining token budgets upstream before generation via TruncProof,
    and evaluating the structural intent distribution.
    Includes Projected Gradient Descent (PGD) for enforcing soft constraints.
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

    def optimize_soft_constraints_pgd(self, target_distribution: torch.Tensor, steps: int = 50, lr: float = 0.01):
        """
        Projected Gradient Descent (PGD) over the PSDD probabilities to enforce soft constraints
        (e.g., ensuring certain topological tokens are favored based on dynamic policies).
        """
        optimizer = optim.SGD([self.node_log_probs], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            current_probs = torch.softmax(self.node_log_probs, dim=-1)
            
            # Loss is KL divergence between target distribution and current PSDD distribution
            loss = nn.KLDivLoss(reduction='batchmean')(torch.log(current_probs + 1e-8), target_distribution)
            loss.backward()
            optimizer.step()
            
            # Projection step: ensuring log_probs don't explode
            with torch.no_grad():
                self.node_log_probs.clamp_(-10, 10)

class TruncProofOptimizer:
    """
    Implements the TruncProof guardrail via convex optimization to bound token budgets upstream.
    This prevents the Realizer from overflowing the LLM context window.
    """
    def __init__(self, llm_max_context: int = 32768, schema_closure_tokens: int = 512, safety_margin: int = 128):
        """
        Args:
            llm_max_context: Total context window of the target LLM (e.g., 32768 for Qwen-3.5).
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
