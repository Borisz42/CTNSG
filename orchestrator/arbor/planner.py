import torch
import torch.nn as nn
from typing import List, Dict, Any

class ArborCritic(nn.Module):
    """
    Critic agent for the Arbor Checks-and-Balances system.
    Evaluates the proposed sub-task DAG to prevent unilateral planning failures.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.evaluator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, structural_intent: torch.Tensor, semantic_intent: torch.Tensor) -> torch.Tensor:
        """
        Returns a confidence score [0, 1] for the proposed decoupling.
        """
        combined = torch.cat([structural_intent, semantic_intent], dim=-1)
        return self.evaluator(combined)

class ArborPlanner(nn.Module):
    """
    Arbor Task-Decoupled Planning.
    Decomposes the global task into a strict DAG of sub-tasks and explicitly 
    separates the structural intent (topology) from the semantic content.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.structural_proj = nn.Linear(input_dim, hidden_dim)
        self.semantic_proj = nn.Linear(input_dim, hidden_dim)
        self.critic = ArborCritic(hidden_dim)
        
    def decouple_plan(self, global_intent_embedding: torch.Tensor, max_retries: int = 3) -> Dict[str, Any]:
        """
        Splits the intent into structural requirements for GVT/RelDiT 
        and semantic requirements for the LLM Realizer.
        Uses the Critic to ensure the plan is valid.
        """
        for attempt in range(max_retries):
            # Propose decoupling (with some noise if retrying)
            noise = torch.randn_like(global_intent_embedding) * 0.1 * attempt
            noisy_intent = global_intent_embedding + noise
            
            structural_intent = self.structural_proj(noisy_intent)
            semantic_intent = self.semantic_proj(noisy_intent)
            
            # Critic evaluation
            confidence = self.critic(structural_intent, semantic_intent)
            
            if confidence.item() > 0.8 or attempt == max_retries - 1:
                return {
                    "structural_intent": structural_intent,
                    "semantic_intent": semantic_intent,
                    "confidence": confidence.item(),
                    "attempts": attempt + 1
                }
                
        # Should not be reached due to fallback above, but added for safety
        return {}

    def generate_subtask_dag(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a directed acyclic graph (DAG) of sub-tasks.
        (Mocked functional fallback for Kaggle compatibility)
        """
        return [
            {"task_id": "t1", "type": "retrieve_context", "depends_on": []},
            {"task_id": "t2", "type": "generate_topology", "depends_on": ["t1"]},
            {"task_id": "t3", "type": "realize_text", "depends_on": ["t2"]},
            {"task_id": "t4", "type": "validate_output", "depends_on": ["t3"]}
        ]
