import torch
import gradio as gr
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os

# Ensure local modules are reachable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from orchestrator.arbor.planner import ArborPlanner
from realizer.realizer import CTNSGRealizer
from contracts.graph_schema import DiscourseGraph, SemanticNode
from macroplanner.gvt.model import GraphVQTransformer
from macroplanner.reldit.model import RelDiT
from realizer.realizer import GreatGrammaLogitsProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading CTNSG Models...")
# Initialize GVT and RelDiT
gvt = GraphVQTransformer(in_channels=256, hidden_channels=256, num_embeddings=64, num_quantizers=4).to(device)
reldit = RelDiT(vocab_size=65, d_model=256).to(device)

export_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ctnsg_export')
if os.path.exists(os.path.join(export_dir, 'gvt_weights.pt')):
    gvt.load_state_dict(torch.load(os.path.join(export_dir, 'gvt_weights.pt'), map_location=device))
    print("Loaded gvt_weights.pt")
if os.path.exists(os.path.join(export_dir, 'reldit_weights.pt')):
    reldit.load_state_dict(torch.load(os.path.join(export_dir, 'reldit_weights.pt'), map_location=device))
    print("Loaded reldit_weights.pt")

# Initialize Realizer LLM
print("Loading Base LLM into Realizer...")
realizer = CTNSGRealizer()
llm = realizer.llm
tokenizer = realizer.tokenizer

print("Loading LoRA SFT Weights for Arbor TDP...")
if os.path.exists(os.path.join(export_dir, 'arbor_lora_weights')):
    from peft import PeftModel
    llm = PeftModel.from_pretrained(llm, os.path.join(export_dir, 'arbor_lora_weights'))
    print("Loaded Arbor LoRA adapters.")

print("Loading Sentence Transformer & Projection...")
from sentence_transformers import SentenceTransformer
sent_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
encoder_projection = torch.nn.Linear(384, 256).to(device)
if os.path.exists(os.path.join(export_dir, 'encoder_projection_weights.pt')):
    encoder_projection.load_state_dict(torch.load(os.path.join(export_dir, 'encoder_projection_weights.pt'), map_location=device))
    print("Loaded 384->256 semantic projection layer.")

arbor_schema = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"}
                },
                "required": ["id", "label"]
            }
        },
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
planner = ArborPlanner(llm=llm, tokenizer=tokenizer, grammar_processor=arbor_processor)

print("Models loaded successfully.")

def process_query(user_query, reldit_toggle, node_count):
    """
    Executes the full CTNSG pipeline for the UI.
    """
    yield (
        "### 🗺️ Planning Graph\n\n⏳ *Orchestrating tasks...*", 
        "### 🧠 Reasoning\n\n⏳ *Waiting for topology...*", 
        "### ✨ Formatted Final Answer\n\n⏳ *Waiting for generation...*"
    )
    
    start_time = time.time()
    
    # 1. Orchestrator
    torch.manual_seed(hash(user_query) % 10000)
    
    if reldit_toggle:
        # Phase 1: Pure Topological Generation (RelDiT)
        with torch.no_grad():
            reldit.eval()
            gen_tokens = reldit.generate(batch_size=1, seq_len=node_count, device=device, use_critic=True)
            curr_tokens = gen_tokens[0].tolist()
            
        unique_edges = set()
        skeleton_edges = []
        for j in range(len(curr_tokens) // 2):
            edge = (f"n_{curr_tokens[2*j]}", f"n_{curr_tokens[2*j+1]}")
            if edge[0] != edge[1] and edge not in unique_edges:  # prevent self-loops and duplicates
                unique_edges.add(edge)
                skeleton_edges.append({"source": edge[0], "target": edge[1], "relation": "depends_on"})
        
        unique_nodes = list(set(f"n_{t}" for t in curr_tokens))
        num_skeleton_nodes = len(unique_nodes)
        
        # Phase 2: Semantic Attachment (Arbor)
        subtasks = planner.generate_subtask_dag(user_query, skeleton_edges=skeleton_edges, unique_nodes=unique_nodes)
    else:
        # Standard Orchestrator Flow
        subtasks = planner.generate_subtask_dag(user_query)
    
    # Build Mermaid DAG
    mermaid_lines = ["### 🗺️ Planning Graph", "", "```mermaid", "graph TD"]
    for i, task in enumerate(subtasks):
        tid = task.get("task_id", f"t{i}")
        ttype = task.get("type", "Task").replace('"', "'")
        mermaid_lines.append(f'    {tid}["{ttype}"]')
        for dep in task.get("depends_on", []):
            mermaid_lines.append(f'    {dep} --> {tid}')
    mermaid_lines.append("```")
    dag_mermaid = "\n".join(mermaid_lines)
    
    # 2. Macroplanner
    num_nodes = max(1, len(subtasks))
    task_descriptions = [task.get("type", "unknown task") for task in subtasks]
    if len(task_descriptions) == 0:
        task_descriptions = ["default task"]
        
    with torch.no_grad():
        raw_embeddings = sent_model.encode(task_descriptions, convert_to_tensor=True).to(device)
        val_features = encoder_projection(raw_embeddings) # [num_nodes, 256]
    
    edge_sources = []
    edge_targets = []
    task_id_to_idx = {task.get("task_id", f"t{i}"): i for i, task in enumerate(subtasks)}
    for i, task in enumerate(subtasks):
        for dep in task.get("depends_on", []):
            if dep in task_id_to_idx:
                edge_sources.append(task_id_to_idx[dep])
                edge_targets.append(i)
                
    if edge_sources:
        val_edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long).to(device)
    else:
        val_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    
    with torch.no_grad():
        gvt.eval()
        reldit.eval()
        out = gvt(val_features, val_edge_index)
        discrete_indices = out['discrete_tokens']
        
    # 3. Realizer
    nodes = [SemanticNode(f"n{i}", f"Task_{i}", int(discrete_indices[min(i, len(discrete_indices)-1)][0].item()) if len(discrete_indices) > 0 else 0) for i in range(val_features.size(0))]
    graph = DiscourseGraph("run1", nodes, [])
    
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "markdown_output": {"type": "string"}
        },
        "required": ["reasoning", "markdown_output"]
    }
    import json
    import re
    
    reasoning = "### 🧠 Reasoning\n\n"
    md_out = "### ✨ Formatted Final Answer\n\n"
    partial_output = ""
    
    for res in realizer.generate_stream(graph, schema, context_lines=5, prompt=user_query, graph_features=out['z_q']):
        partial_output = res['text']
        
        try:
            out_data = json.loads(partial_output)
            raw_reasoning = out_data.get("reasoning", "")
            raw_md_out = out_data.get("markdown_output", "")
            
            if isinstance(raw_reasoning, str):
                raw_reasoning = raw_reasoning.replace('\\n', '\n')
            if isinstance(raw_md_out, str):
                raw_md_out = raw_md_out.replace('\\n', '\n')
                
            reasoning = "### 🧠 Reasoning\n\n" + str(raw_reasoning)
            md_out = "### ✨ Formatted Final Answer\n\n" + str(raw_md_out)
            yield dag_mermaid, reasoning, md_out
            
        except Exception:
            reasoning = "### 🧠 Reasoning\n\n"
            md_out = "### ✨ Formatted Final Answer\n\n"
            
            def decode_partial(s):
                if s.endswith('\\') and not s.endswith('\\\\'):
                    s = s[:-1]
                for suffix in ['"', '\\"', '""']:
                    try:
                        res = json.loads('"' + s + suffix)
                        if isinstance(res, str):
                            return res.replace('\\n', '\n')
                    except Exception:
                        pass
                return s.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            
            r_match = re.search(r'"reasoning":\s*"((?:[^"\\]|\\.)*)', partial_output)
            if r_match:
                reasoning += decode_partial(r_match.group(1)) + " ▌"
            
            m_match = re.search(r'"markdown_output":\s*"((?:[^"\\]|\\.)*)', partial_output)
            if m_match:
                md_out += decode_partial(m_match.group(1)) + " ▌"
                
            if not r_match and not m_match:
                reasoning += "*(Generating JSON...)* ▌"
                md_out += f"```json\n{partial_output}\n```"
                
            yield dag_mermaid, reasoning, md_out

    # Final cleanup yield after stream finishes
    for stop_str in ["<|im_end|>", "<|end|>"]:
        if stop_str in partial_output:
            partial_output = partial_output.split(stop_str)[0]
            
    partial_output = re.sub(r',\s*\]', ']', partial_output)
    partial_output = re.sub(r',\s*\}', '}', partial_output)
    
    try:
        out_data = json.loads(partial_output)
        raw_reasoning = out_data.get("reasoning", "")
        raw_md_out = out_data.get("markdown_output", "")
        if isinstance(raw_reasoning, str):
            raw_reasoning = raw_reasoning.replace('\\n', '\n')
        if isinstance(raw_md_out, str):
            raw_md_out = raw_md_out.replace('\\n', '\n')
            
        reasoning = "### 🧠 Reasoning\n\n" + str(raw_reasoning)
        md_out = "### ✨ Formatted Final Answer\n\n" + str(raw_md_out)
    except Exception:
        reasoning = reasoning.replace(" ▌", "")
        md_out = md_out.replace(" ▌", "")
        
    yield dag_mermaid, reasoning, md_out
            
# Gradio Interface
with gr.Blocks(title="CTNSG Framework Inference Harness") as demo:
    gr.Markdown("# Canonical Tractable Neuro-Symbolic Generation")
    gr.Markdown("Visualize the multi-module neuro-symbolic pipeline execution.")
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="User Intent / Query", placeholder="Enter a complex logical request...", lines=3)
            reldit_toggle = gr.Checkbox(label="Enable Massive Structural Blueprinting (RelDiT)", value=False)
            node_count_slider = gr.Slider(minimum=1, maximum=100, step=1, value=54, label="RelDiT Node Count (Skeleton Size)")
            submit_btn = gr.Button("Generate with CTNSG", variant="primary")
            
        with gr.Column(scale=2):
            dag_visual = gr.Markdown(label="Sub-task DAG (Macroplanner Topology)")
            reasoning_output = gr.Markdown(label="Supervisor Reasoning")
            markdown_output = gr.Markdown(label="Final Syntactic Realization (L1 Validated)")
            
    submit_btn.click(fn=process_query, inputs=[query_input, reldit_toggle, node_count_slider], outputs=[dag_visual, reasoning_output, markdown_output])

if __name__ == "__main__":
    print("Starting CTNSG UI Harness...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Monochrome())
