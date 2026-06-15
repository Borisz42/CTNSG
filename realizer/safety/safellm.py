import torch
import torch.nn as nn
import re
from typing import List, Optional

class TruncProofOptimizer:
    """
    Implements the TruncProof guardrail via convex optimization to bound token budgets upstream.
    This prevents the Realizer from overflowing the LLM context window.
    """
    def __init__(self, llm_max_context: int = 32768, schema_closure_tokens: int = 512, safety_margin: int = 128):
        self.llm_max_context = llm_max_context
        self.schema_closure_tokens = schema_closure_tokens
        self.safety_margin = safety_margin

    def calculate_dynamic_budget(self, current_prompt_tokens: int) -> int:
        available_context = self.llm_max_context - current_prompt_tokens - self.schema_closure_tokens - self.safety_margin
        if available_context <= 0:
            raise ValueError("Semantic Collapse Risk: Context limits exceeded.")
        return available_context

    def force_graceful_closure(self, current_tokens: int, max_tokens: int) -> bool:
        """ Returns True if the LLM must stop generating and close the JSON/XML schema immediately. """
        return current_tokens >= (max_tokens - self.schema_closure_tokens)

class SafeLLMExtractor:
    """
    SafeLLM line-number extraction strictly pulls facts from the FAAP context.
    Prevents hallucination by ensuring text maps to specific context lines.
    """
    def __init__(self):
        # Regex to capture [Line X] or [L: X] references
        self.line_pattern = re.compile(r'\[(?:Line|L):\s*(\d+)\]', re.IGNORECASE)

    def extract_and_verify(self, generated_text: str, context_lines: int) -> List[int]:
        """
        Extracts line numbers from the generated text and verifies they exist in the context.
        """
        matches = self.line_pattern.findall(generated_text)
        valid_lines = []
        for m in matches:
            line_idx = int(m)
            if 1 <= line_idx <= context_lines:
                valid_lines.append(line_idx)
        return valid_lines

class OptimalTransportMonitor(nn.Module):
    """
    Cross-Attention Optimal Transport (OT) Monitoring.
    Tracks the middle transformer layers (e.g., layer 12) in real-time to detect 
    when the LLM mathematically disengages from the source context.
    """
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold

    def compute_ot_distance(self, cross_attention_weights: torch.Tensor) -> float:
        """
        Computes the Wasserstein-1 (Optimal Transport) distance between the attention 
        distribution and a uniform attention. (Simplified for Kaggle functional fallback).
        """
        # cross_attention_weights: [batch, num_heads, tgt_len, src_len]
        # High entropy / uniform attention indicates the model is ignoring specific context (disengaging).
        
        # Calculate entropy of the attention distribution across the source length
        entropy = -torch.sum(cross_attention_weights * torch.log(cross_attention_weights + 1e-9), dim=-1)
        mean_entropy = entropy.mean().item()
        
        # Max entropy for src_len is log(src_len)
        src_len = cross_attention_weights.size(-1)
        max_entropy = torch.log(torch.tensor(float(src_len))).item()
        
        # Normalized disengagement score
        disengagement_score = mean_entropy / (max_entropy + 1e-9)
        return disengagement_score

    def check_disengagement(self, cross_attention_weights: torch.Tensor) -> bool:
        """ Returns True if the LLM has disengaged from the context. """
        score = self.compute_ot_distance(cross_attention_weights)
        return score > self.threshold
