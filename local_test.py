import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.append(os.path.abspath('.'))

from macroplanner.gvt.model import GraphVQTransformer
from macroplanner.reldit.model import RelDiT
from orchestrator.arbor.planner import ArborPlanner
from realizer.realizer import CTNSGRealizer
from contracts.graph_schema import DiscourseGraph, SemanticNode, SemanticEdge

class GVTWrapper(nn.Module):
    """Wrapper to bypass DataParallel's edge_index splitting limitation by absorbing empty edges."""
    def __init__(self, gvt):
        super().__init__()
        self.gvt = gvt
    
    def forward(self, nodes):
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=nodes.device)
        return self.gvt(nodes, empty_edges)

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Training on {device} with {num_gpus} GPUs")
    
    if num_gpus > 0:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 22:
            base_bs = 8
        elif vram_gb >= 14:
            base_bs = 4
        elif vram_gb >= 7:
            base_bs = 2
        else:
            base_bs = 1
        batch_size = base_bs * num_gpus
    else:
        vram_gb = 0
        batch_size = 2
        
    print(f"Detected VRAM per GPU: {vram_gb:.1f} GB. Dynamically set global batch size to: {batch_size}")
    max_seq = 1024
    
    print("\n--- 1. Load Preprocessed Supervisor Datasets (Module 2 & 4) ---")
    def load_pt(path):
        return torch.load(path) if os.path.exists(path) else []
    
    faap_data = []
    if os.path.exists('processed_data/faap_instructions_full.jsonl'):
        with open('processed_data/faap_instructions_full.jsonl', 'r', encoding='utf-8') as f:
            faap_data = [json.loads(line) for line in f]
    print(f"Loaded {len(faap_data)} FAAP instruction pairs.")
    
    sdrt_graphs = load_pt('processed_data/sdrt_graphs_full.pt')
    arbor_graphs = load_pt('processed_data/arbor_graphs_full.pt')
    verification_graphs = load_pt('processed_data/verification_graphs_full.pt')
    
    print(f"Loaded {len(sdrt_graphs)} SDRT Discourse Graphs.")
    print(f"Loaded {len(arbor_graphs)} Arbor TDP True DAGs.")
    print(f"Loaded {len(verification_graphs)} SAIGuard/Brick Interaction Graphs.")

    print("\n--- 2. Arbor Orchestrator Agentic SFT (Module 2) ---")
    print("Running a quick local LoRA SFT loop (2 samples) using train_arbor_sft...")
    from orchestrator.arbor.train import train_arbor_sft
    
    arbor_model, tokenizer, proj_layer = train_arbor_sft(
        model_name="unsloth/Phi-4-mini-instruct",
        arbor_graphs=arbor_graphs[:2],
        device=str(device),
        epochs=1,
        lr=2e-5,
        output_dir="ctnsg_export"
    )
    
    # Instantiate the agentic planner (it falls back to dummy inference without a grammar processor during the test script, just verifying API signature)
    planner = ArborPlanner(llm=arbor_model, tokenizer=tokenizer)
    subtasks = planner.generate_subtask_dag("Generate test task")
    print("Planner Inference DAG Mock:", subtasks)

    # Free up VRAM
    del arbor_model
    del planner
    torch.cuda.empty_cache()
    
    print("\n--- 3. GVT Training Loop ---")
    gvt_base = GraphVQTransformer(in_channels=256, hidden_channels=256, num_embeddings=64, num_quantizers=4).to(device)
    
    if num_gpus > 1:
        gvt = nn.DataParallel(GVTWrapper(gvt_base))
    else:
        gvt = GVTWrapper(gvt_base)
        
    gvt_optimizer = optim.AdamW(gvt.parameters(), lr=3e-4)
    
    mock_node_chunks = [torch.randn(max_seq, 256).to(device) for _ in range(batch_size)]
    mock_nodes_batched = torch.stack(mock_node_chunks, dim=0)
    
    assert mock_nodes_batched.shape == (batch_size, max_seq, 256), "Batching error!"
    
    for epoch in range(1):
        gvt_optimizer.zero_grad()
        out = gvt(mock_nodes_batched)
        quantized_latents = out['z_q']
        vq_loss = out['commit_loss'].mean()
        discrete_indices = out['discrete_tokens']
        
        recon_loss = nn.MSELoss()(quantized_latents, mock_nodes_batched)
        total_loss = recon_loss + vq_loss
        total_loss.backward()
        gvt_optimizer.step()
        print(f"Epoch {epoch+1} | GVT Batched Loss: {total_loss.item():.4f}")
        
    print("\n--- 4. RelDiT Training Loop ---")
    reldit_base = RelDiT(vocab_size=65, d_model=256).to(device)
    
    if num_gpus > 1:
        reldit = nn.DataParallel(reldit_base)
    else:
        reldit = reldit_base
        
    reldit_optimizer = optim.AdamW(reldit.parameters(), lr=1e-4)
    
    for epoch in range(1):
        reldit_optimizer.zero_grad()
        tokens = discrete_indices[:, :, 0] + 1
        loss = reldit(tokens)
        loss = loss.mean()
        loss.backward()
        reldit_optimizer.step()
        print(f"Epoch {epoch+1} | RelDiT Batched Loss: {loss.item():.4f}")
        
    # Free up VRAM before loading the realizer
    del gvt
    del reldit
    del gvt_optimizer
    del reldit_optimizer
    torch.cuda.empty_cache()
        
    print("\n--- 5. Realizer Inference Pipeline ---")
    realizer = CTNSGRealizer(vocab_size=3200, hidden_dim=256, model_name="unsloth/Phi-4-mini-instruct")
    inference_graph = DiscourseGraph(
        graph_id="infer_001",
        nodes=[
            SemanticNode(node_id="n1", concept="System", vq_index=12),
            SemanticNode(node_id="n2", concept="Macroplanner", vq_index=45)
        ],
        edges=[
            SemanticEdge(source_id="n1", target_id="n2", relation_type="triggers")
        ]
    )
    schema = {"type": "object", "properties": {"output": {"type": "string"}}}
    result = realizer.generate(inference_graph, schema, context_lines=5)
    print("\n=== Realizer Final Output ===")
    print(result['text'])
    
if __name__ == "__main__":
    run_test()
