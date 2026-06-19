import sys
import os
import torch
import torch.nn as nn
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath('.'))

from macroplanner.gvt.model import GraphVQTransformer
from orchestrator.arbor.planner import ArborPlanner
from realizer.realizer import CTNSGRealizer
from contracts.graph_schema import DiscourseGraph, SemanticNode, SemanticEdge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running inference on {device}")

print("\n--- Loading Models ---")
model_name = "unsloth/Phi-4-mini-instruct"

# 1. Load Orchestrator (Arbor LoRA over Base LLM)
print("Loading Arbor_LoRA...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)
# Load PEFT adapter
try:
    arbor_model = PeftModel.from_pretrained(base_model, "ctnsg_export/arbor_lora_weights")
except ValueError:
    print("Warning: arbor_lora_weights not found. Using base model as fallback.")
    arbor_model = base_model

# 2. Load Realizer (Uses the same base LLM)
print("Initializing Realizer...")
realizer = CTNSGRealizer(model_name=model_name, hidden_dim=256, cache_dir="ctnsg_export/.psc_cache")
# We share the base LLM instance for memory efficiency
realizer.llm = base_model
realizer.tokenizer = tokenizer

# 3. Load Sentence Transformer & Projection Layer
print("Loading Sentence Transformer & Encoder Projection...")
sent_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
proj_layer = nn.Linear(384, 256).to(device)
if os.path.exists("ctnsg_export/encoder_projection_weights.pt"):
    proj_layer.load_state_dict(torch.load("ctnsg_export/encoder_projection_weights.pt", map_location=device))
else:
    print("Warning: encoder_projection_weights.pt not found. Using untrained weights.")
proj_layer.eval()

# 4. Load GVT
print("Loading GVT...")
gvt = GraphVQTransformer(in_channels=256, hidden_channels=256, num_embeddings=64, num_quantizers=4).to(device)
if os.path.exists("ctnsg_export/gvt_weights.pt"):
    gvt.load_state_dict(torch.load("ctnsg_export/gvt_weights.pt", map_location=device))
else:
    print("Warning: gvt_weights.pt not found. Using untrained weights.")
gvt.eval()

print("\n--- 4-Question Test (Full Pipeline) ---")
questions = [
    "What is the capital of France?",
    "If all fleeps are bloops, and some bloops are zopfs, are all fleeps definitely zopfs?",
    "Write a simple Python function to calculate the factorial of a number.",
    "Write a short 4-line poem about the stars."
]

from realizer.realizer import GreatGrammaLogitsProcessor

arbor_schema = {
    "type": "object",
    "properties": {
        "nodes": {"type": "array", "items": {"type": "string"}},
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"}
                }
            }
        }
    },
    "required": ["nodes", "edges"]
}

arbor_processor = GreatGrammaLogitsProcessor(realizer.grammar, arbor_schema, tokenizer=tokenizer)

# Instantiate the planner using the fine-tuned PEFT model and GCD mask
planner_pipeline = ArborPlanner(llm=arbor_model, tokenizer=tokenizer, grammar_processor=arbor_processor)

schema = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "markdown_output": {"type": "string"}
    },
    "required": ["reasoning", "markdown_output"]
}

for i, q in enumerate(questions, 1):
    print(f"\nQuestion {i}: {q}")
    
    # 1. Orchestrator (Arbor TDP)
    subtasks = planner_pipeline.generate_subtask_dag(q)
    print(" -> Generated DAG:", subtasks)
    
    # 2. Macroplanner (RelDiT / GVT)
    task_descriptions = [task.get("type", "unknown task") for task in subtasks]
    if not task_descriptions:
        task_descriptions = ["default task"]
    
    with torch.no_grad():
        raw_embeddings = sent_model.encode(task_descriptions, convert_to_tensor=True).to(device)
        val_features = proj_layer(raw_embeddings)
    
    edge_sources = []
    edge_targets = []
    task_id_to_idx = {task.get("task_id", f"t{idx}"): idx for idx, task in enumerate(subtasks)}
    for idx, task in enumerate(subtasks):
        for dep in task.get("depends_on", []):
            if dep in task_id_to_idx:
                edge_sources.append(task_id_to_idx[dep])
                edge_targets.append(idx)
                
    if edge_sources:
        val_edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long).to(device)
    else:
        val_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        
    with torch.no_grad():
        out = gvt(val_features, val_edge_index)
        discrete_indices = out['discrete_tokens']
        
    # 3. Realizer
    nodes = [SemanticNode(f"n{idx}", f"Task_{idx}", int(discrete_indices[min(idx, len(discrete_indices)-1)][0].item()) if len(discrete_indices) > 0 else 0) for idx in range(val_features.size(0))]
    q_graph = DiscourseGraph(f"q_{i}", nodes, [])
    
    result = realizer.generate(q_graph, schema, context_lines=3, prompt=q, graph_features=out['z_q'])
    print(f" -> Answer {i} Final Output:")
    print(result.get('text', ''))
