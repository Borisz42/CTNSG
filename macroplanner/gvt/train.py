import os
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from macroplanner.gvt.model import GraphVQTransformer

class GVTWrapper(nn.Module):
    def __init__(self, gvt):
        super().__init__()
        self.gvt = gvt
    def forward(self, nodes):
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=nodes.device)
        return self.gvt(nodes, empty_edges)

def train_gvt(
    curriculum_graphs,
    device: str = "cuda",
    epochs: int = 3,
    lr: float = 3e-4,
    in_channels: int = 256,
    hidden_channels: int = 256,
    num_embeddings: int = 64,
    num_quantizers: int = 4,
    max_seq: int = 1024,
    output_dir: str = "ctnsg_export"
):
    """
    Trains the Graph Vector Transformer (GVT) discrete compression model.
    """
    num_gpus = torch.cuda.device_count()
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
    
    gvt_base = GraphVQTransformer(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_embeddings=num_embeddings,
        num_quantizers=num_quantizers
    ).to(device)
    
    if num_gpus > 1:
        gvt = nn.DataParallel(GVTWrapper(gvt_base))
    else:
        gvt = GVTWrapper(gvt_base)
        
    gvt_optimizer = optim.AdamW(gvt.parameters(), lr=lr)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting GVT Training Loop (Batch Size: {batch_size})...")
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss_epoch = 0.0
        batched_nodes = []
        
        for graph in curriculum_graphs:
            nodes_full = graph['nodes'].to(device)
            for i in range(0, nodes_full.shape[0], max_seq):
                nodes = nodes_full[i : i + max_seq]
                if nodes.size(0) < max_seq:
                    pad_size = max_seq - nodes.size(0)
                    nodes = torch.cat([nodes, torch.zeros(pad_size, nodes.size(1), device=device)], dim=0)
                    
                batched_nodes.append(nodes)
                
                if len(batched_nodes) == batch_size:
                    nodes_tensor = torch.stack(batched_nodes, dim=0)
                    gvt_optimizer.zero_grad()
                    out = gvt(nodes_tensor)
                    quantized_latents = out['z_q']
                    vq_loss = out['commit_loss'].mean()
                    recon_loss = nn.MSELoss()(quantized_latents, nodes_tensor)
                    total_loss = recon_loss + vq_loss
                    total_loss.backward()
                    gvt_optimizer.step()
                    total_loss_epoch += total_loss.item()
                    batched_nodes = []
                    
        if len(batched_nodes) > 0:
            nodes_tensor = torch.stack(batched_nodes, dim=0)
            gvt_optimizer.zero_grad()
            out = gvt(nodes_tensor)
            quantized_latents = out['z_q']
            vq_loss = out['commit_loss'].mean()
            recon_loss = nn.MSELoss()(quantized_latents, nodes_tensor)
            total_loss = recon_loss + vq_loss
            total_loss.backward()
            gvt_optimizer.step()
            total_loss_epoch += total_loss.item()
            
        epoch_duration = time.time() - epoch_start
        avg_loss = total_loss_epoch / max(1, len(curriculum_graphs) // batch_size)
        eta_seconds = epoch_duration * (epochs - (epoch + 1))
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Epoch {epoch+1}/{epochs} | GVT Loss: {avg_loss:.4f} | Time: {epoch_duration:.1f}s | ETA: {eta_str}")
        
    print("Saving GVT weights...")
    os.makedirs(output_dir, exist_ok=True)
    if num_gpus > 1:
        torch.save(gvt.module.gvt.state_dict(), os.path.join(output_dir, "gvt_weights.pt"))
    else:
        torch.save(gvt.gvt.state_dict(), os.path.join(output_dir, "gvt_weights.pt"))
        
    return gvt_base, gvt
