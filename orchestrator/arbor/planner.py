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
        
    def generate_subtask_dag(self, user_query: str, skeleton_edges: List[Dict[str, Any]] = None, unique_nodes: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generates a directed acyclic graph (DAG) of sub-tasks by prompting the LLM
        with O(1) Grammar-Constrained Decoding (GCD) via the logits processor.
        If skeleton_edges are provided, acts as a semantic router.
        """
        if self.llm is None or self.tokenizer is None:
            # Fallback for testing without LLM loaded
            return [
                {"task_id": "t1", "type": "retrieve_context", "depends_on": []},
                {"task_id": "t2", "type": "generate_topology", "depends_on": ["t1"]},
                {"task_id": "t3", "type": "realize_text", "depends_on": ["t2"]},
                {"task_id": "t4", "type": "validate_output", "depends_on": ["t3"]}
            ]
            
        if unique_nodes is not None:
            sys_msg = (
                f"You are the Arbor Semantic Router. I have a topology with the following Node IDs: {json.dumps(unique_nodes)}.\n"
                "Based on the user's prompt, assign appropriate semantic labels to these node IDs. "
                "Output the semantic mapping in valid JSON conforming to the following schema:\n"
                "{\n"
                "  \"nodes\": [{\"id\": \"node_id\", \"label\": \"Task Description\"}, ...],\n"
                "  \"edges\": []\n"
                "}\n"
                "You must output an empty 'edges' array because I already have the topology."
            )
        else:
            sys_msg = (
                "You are the Arbor Supervisor. Decompose the user's task into a strict JSON DAG "
                "conforming to the following schema:\n"
                "{\n"
                "  \"nodes\": [{\"id\": \"task_id_1\", \"label\": \"Task Description\"}, ...],\n"
                "  \"edges\": [\n"
                "    {\"source\": \"task_id_1\", \"target\": \"task_id_2\", \"relation\": \"depends_on\"}\n"
                "  ]\n"
                "}\n"
                "Do not use tuples like (a, b) in edges. Use objects with \"source\", \"target\", and \"relation\" keys."
            )
            
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_query}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        # Apply O(1) GREATGRAMMA JSON constraint mask
        logits_processor = []
        if self.grammar_processor is not None:
            from realizer.safety.safellm import TruncProofOptimizer
            trunc_proof = TruncProofOptimizer(llm_max_context=32768, schema_closure_tokens=64, safety_margin=32)
            prompt_tokens = inputs.input_ids.shape[1]
            try:
                dynamic_budget = trunc_proof.calculate_dynamic_budget(prompt_tokens)
            except ValueError:
                dynamic_budget = 64
            max_new_tokens = min(4096, dynamic_budget)
            
            # Configure grammar processor dynamically for this run
            self.grammar_processor.trunc_proof_optimizer = trunc_proof
            self.grammar_processor.dynamic_budget = max_new_tokens
            self.grammar_processor.current_step = 0
            self.grammar_processor.current_state_id = 0
            logits_processor.append(self.grammar_processor)
        else:
            max_new_tokens = 4096
            
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
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        for stop_str in ["<|im_end|>", "<|end|>"]:
            if stop_str in response:
                response = response.split(stop_str)[0]
                
        # Clean trailing commas (common LLM JSON hallucination)
        import re
        response = re.sub(r',\s*\]', ']', response)
        response = re.sub(r',\s*\}', '}', response)
        
        try:
            dag = json.loads(response)
            subtasks = []
            if isinstance(dag, dict) and "nodes" in dag:
                if unique_nodes and skeleton_edges:
                    assembled_subtasks = []
                    generated_labels = {}
                    
                    for node in dag.get("nodes", []):
                        if isinstance(node, dict):
                            generated_labels[node.get("id")] = node.get("label", "Intermediate Task")
                            
                    for nid in unique_nodes:
                        assembled_subtasks.append({
                            "task_id": nid,
                            "type": generated_labels.get(nid, "Intermediate Task"),
                            "depends_on": []
                        })
                        
                    for edge in skeleton_edges:
                        source = edge.get("source")
                        target = edge.get("target")
                        for st in assembled_subtasks:
                            if st["task_id"] == target and source not in st["depends_on"]:
                                st["depends_on"].append(source)
                                
                    return assembled_subtasks
                else:
                    for node in dag.get("nodes", []):
                        if isinstance(node, dict):
                            task_id = node.get("id", "unknown")
                            task_type = node.get("label", node.get("id", "unknown"))
                        else:
                            task_id = node
                            task_type = node
                        subtasks.append({
                            "task_id": task_id,
                            "type": task_type,
                            "depends_on": []
                        })
                    for edge in dag.get("edges", []):
                        source = edge.get("source")
                        target = edge.get("target")
                        for st in subtasks:
                            if st["task_id"] == target:
                                st["depends_on"].append(source)
                                
                    return subtasks
            elif isinstance(dag, list):
                return dag
            else:
                return [dag]
        except json.JSONDecodeError as e:
            # Due to the GCD constraint, this block shouldn't trigger unless schema compilation failed or token limit hit
            print(f"Warning: Arbor LLM hallucinated invalid JSON despite GCD: {e}. Falling back to default DAG.")
            print("Raw response was:", repr(response))
            
            # If we have a topological skeleton, deterministically assemble it with fallback labels
            if unique_nodes and skeleton_edges:
                assembled_subtasks = []
                for nid in unique_nodes:
                    assembled_subtasks.append({
                        "task_id": nid,
                        "type": "Intermediate Task (Fallback)",
                        "depends_on": []
                    })
                for edge in skeleton_edges:
                    source = edge.get("source")
                    target = edge.get("target")
                    for st in assembled_subtasks:
                        if st["task_id"] == target and source not in st["depends_on"]:
                            st["depends_on"].append(source)
                return assembled_subtasks
            
            # Standard flow fallback
            return [
                {"task_id": "t1", "type": "retrieve_context", "depends_on": []},
                {"task_id": "t2", "type": "generate_topology", "depends_on": ["t1"]},
                {"task_id": "t3", "type": "semantic_routing", "depends_on": ["t2"]},
                {"task_id": "t4", "type": "validate_output", "depends_on": ["t3"]}
            ]
