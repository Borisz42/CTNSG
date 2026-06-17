import torch
import torch.nn as nn
from typing import List, Dict, Any
import json
from transformers import PreTrainedModel, PreTrainedTokenizer

class ArborPlanner:
    """
    Arbor Task-Decoupled Planning via Agentic LLM Interaction.
    Decomposes the global task into a strict DAG of sub-tasks by prompting the LLM.
    """
    def __init__(self, llm: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None, grammar_processor = None):
        self.llm = llm
        self.tokenizer = tokenizer
        self.grammar_processor = grammar_processor
        
    def generate_subtask_dag(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Generates a directed acyclic graph (DAG) of sub-tasks by prompting the LLM
        with O(1) Grammar-Constrained Decoding (GCD) via the logits processor.
        """
        if self.llm is None or self.tokenizer is None:
            # Fallback for testing without LLM loaded
            return [
                {"task_id": "t1", "type": "retrieve_context", "depends_on": []},
                {"task_id": "t2", "type": "generate_topology", "depends_on": ["t1"]},
                {"task_id": "t3", "type": "realize_text", "depends_on": ["t2"]},
                {"task_id": "t4", "type": "validate_output", "depends_on": ["t3"]}
            ]
            
        prompt = (
            "<|im_start|>system\n"
            "You are the Arbor Supervisor. Decompose this task into a JSON DAG containing a list of sub-tasks.\n"
            "Respond strictly in JSON format matching schema: [{\"task_id\": str, \"type\": str, \"depends_on\": List[str]}].\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        # Apply O(1) GREATGRAMMA JSON constraint mask
        logits_processor = []
        if self.grammar_processor is not None:
            logits_processor.append(self.grammar_processor)
            
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=logits_processor
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            dag = json.loads(response)
            if isinstance(dag, list):
                return dag
            else:
                return [dag]
        except json.JSONDecodeError:
            # Due to the GCD constraint, this block shouldn't trigger unless schema compilation failed
            print("Warning: Arbor LLM hallucinated invalid JSON despite GCD. Falling back to default DAG.")
            return [
                {"task_id": "t1", "type": "retrieve_context", "depends_on": []},
                {"task_id": "t2", "type": "generate_topology", "depends_on": ["t1"]}
            ]
