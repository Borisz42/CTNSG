import os
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from macroplanner.reldit.model import RelDiT

def cache_graphs(graphs, gvt_model, bs, max_seq, device):
    gvt_model.eval()
    all_tokens = []
    batched_nodes = []
    with torch.no_grad():
        for graph in graphs:
            nodes_full = graph['nodes'].to(device)
            for i in range(0, nodes_full.shape[0], max_seq):
                nodes = nodes_full[i : i + max_seq]
                if nodes.size(0) < max_seq:
                    nodes = torch.cat([nodes, torch.zeros(max_seq - nodes.size(0), nodes.size(1), device=device)], dim=0)
                batched_nodes.append(nodes)
                if len(batched_nodes) == bs:
                    nodes_tensor = torch.stack(batched_nodes, dim=0)
                    out = gvt_model(nodes_tensor)
                    tokens = out['discrete_tokens'][:, :, 0]
                    all_tokens.append(tokens.cpu())
                    batched_nodes = []
        if len(batched_nodes) > 0:
            nodes_tensor = torch.stack(batched_nodes, dim=0)
            out = gvt_model(nodes_tensor)
            tokens = out['discrete_tokens'][:, :, 0]
            all_tokens.append(tokens.cpu())
    if all_tokens:
        return torch.cat(all_tokens, dim=0)
    return torch.empty((0, max_seq), dtype=torch.long)

def train_reldit(
    curriculum_graphs,
    gvt,
    device: str = "cuda",
    epochs: int = 100,
    lr: float = 3e-4,
    vocab_size: int = 65,
    d_model: int = 256,
    max_seq: int = 1024,
    output_dir: str = "ctnsg_export"
):
    """
    Trains the RelDiT relational diffusion model on curriculum graphs.
    """
    val_split_size = min(200, len(curriculum_graphs) // 10)
    train_graphs = curriculum_graphs[:-val_split_size]
    val_graphs = curriculum_graphs[-val_split_size:]
    
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
        batch_size = 2
        
    print("Pre-computing all discrete tokens...")
    eval_batch_size = batch_size * 2
    cached_train_dataset = cache_graphs(train_graphs, gvt, eval_batch_size, max_seq, device)
    cached_val_dataset = cache_graphs(val_graphs, gvt, eval_batch_size, max_seq, device)
    cached_train_dataset = cached_train_dataset + 1
    cached_val_dataset = cached_val_dataset + 1
    print(f"Caching complete! Train shape: {cached_train_dataset.shape}, Val shape: {cached_val_dataset.shape}")
    print(f"Data split: {len(train_graphs)} training graphs, {len(val_graphs)} validation graphs.")
    
    reldit_base = RelDiT(vocab_size=vocab_size, d_model=d_model).to(device)
    if num_gpus > 1:
        reldit = nn.DataParallel(reldit_base)
    else:
        reldit = reldit_base
        
    reldit_optimizer = optim.AdamW(reldit.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reldit_optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    best_validity = -1.0
    patience = 20
    patience_counter = 0
    
    print("Caching training tokens for V.U.N Novelty tracking...")
    train_token_cache = set()
    for i in range(len(cached_train_dataset)):
        tokens_np = cached_train_dataset[i].numpy().tobytes()
        train_token_cache.add(tokens_np)
        
    # Dynamic batch size for RelDiT (smaller model, much larger batch size)
    if num_gpus > 0:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 22:
            reldit_base_bs = 64
        elif vram_gb >= 14:
            reldit_base_bs = 32
        elif vram_gb >= 7:
            reldit_base_bs = 16
        else:
            reldit_base_bs = 8
        reldit_batch_size = reldit_base_bs * num_gpus
    else:
        reldit_batch_size = 8
        
    try:
        cached_train_dataset = cached_train_dataset.to(device)
        cached_val_dataset = cached_val_dataset.to(device)
        print("Moved cached datasets entirely to GPU VRAM for maximum speed.")
    except RuntimeError:
        print("Not enough VRAM to keep entire dataset on GPU, leaving on CPU.")
        
    print()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting RelDiT Training Loop (Batch Size: {reldit_batch_size})...")
    reldit_start_time = time.time()
    max_reldit_time = 11 * 3600
    
    for epoch in range(epochs):
        epoch_start = time.time()
        if time.time() - reldit_start_time > max_reldit_time:
            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 11-hour time limit reached! Halting RelDiT training safely.")
            reldit.load_state_dict(torch.load('best_reldit_weights.pt'))
            break
            
        reldit.train()
        total_loss_epoch = 0.0
        indices = torch.randperm(len(cached_train_dataset), device=device if cached_train_dataset.is_cuda else 'cpu')
        
        for i in range(0, len(cached_train_dataset), reldit_batch_size):
            batch_indices = indices[i:i + reldit_batch_size]
            token_batch = cached_train_dataset[batch_indices].to(device)
            
            reldit_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = reldit(token_batch)
                loss = loss.mean()
                
            scaler.scale(loss).backward()
            scaler.step(reldit_optimizer)
            scaler.update()
            
            total_loss_epoch += loss.item()
            
        avg_train_loss = total_loss_epoch / max(1, len(cached_train_dataset) // reldit_batch_size)
        
        reldit.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for i in range(0, len(cached_val_dataset), reldit_batch_size):
                token_batch = cached_val_dataset[i:i + reldit_batch_size].to(device)
                with torch.amp.autocast('cuda'):
                    loss = reldit(token_batch)
                    loss = loss.mean()
                val_loss_epoch += loss.item()
                
        avg_val_loss = val_loss_epoch / max(1, len(cached_val_dataset) // reldit_batch_size)
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start
        eta_seconds = epoch_duration * (epochs - (epoch + 1))
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration:.1f}s | ETA: {eta_str}")
        
        if (epoch + 1) % 5 == 0:
            print(f"  -> Evaluating V.U.N Metrics (Epoch {epoch+1})...")
            N_samples = 10
            gen_seq_len = 1024
            with torch.no_grad():
                gen_tokens = reldit_base.generate(batch_size=N_samples, seq_len=gen_seq_len, device=device, use_critic=False)
                scaled_logits = reldit_base.transformer(gen_tokens, torch.ones((gen_tokens.size(0),), device=device, dtype=torch.long)) / 1.0
                probs = torch.softmax(scaled_logits, dim=-1)
                pred_probs = torch.gather(probs, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)
                validity = pred_probs.mean().item() * 100.0
                unique_seqs = set([tuple(seq.tolist()) for seq in gen_tokens])
                uniqueness = (len(unique_seqs) / N_samples) * 100.0
                novel_count = sum(1 for seq in unique_seqs if torch.tensor(seq, dtype=torch.long).numpy().tobytes() not in train_token_cache)
                novelty = (novel_count / max(1, len(unique_seqs))) * 100.0
            print(f"  -> [V.U.N] Validity: {validity:.1f}% | Uniqueness: {uniqueness:.1f}% | Novelty: {novelty:.1f}%")
            if validity > best_validity:
                best_validity = validity
                patience_counter = 0
                torch.save(reldit_base.state_dict(), 'best_reldit_weights.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Early stopping! Best Validity: {best_validity:.1f}%")
                    reldit_base.load_state_dict(torch.load('best_reldit_weights.pt'))
                    break
                    
    print("Saving RelDiT weights...")
    os.makedirs(output_dir, exist_ok=True)
    if num_gpus > 1:
        torch.save(reldit.module.state_dict(), os.path.join(output_dir, "reldit_weights.pt"))
    else:
        torch.save(reldit.state_dict(), os.path.join(output_dir, "reldit_weights.pt"))
        
    return reldit_base, reldit, cached_train_dataset, reldit_batch_size

def train_critic(
    reldit_base,
    reldit,
    cached_train_dataset,
    reldit_batch_size,
    device: str = "cuda",
    epochs: int = 10
):
    """
    Trains the post-hoc critic model for iterative denoiser reinforcement.
    """
    print('Freezing the main RelDiT Denoiser parameters...')
    for name, param in reldit_base.named_parameters():
        if 'critic' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    critic_optimizer = optim.AdamW(reldit_base.critic.parameters(), lr=1e-4)
    bce_loss = nn.BCELoss()
    critic_scaler = torch.amp.GradScaler('cuda')
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Critic Training Loop...")
    critic_train_tokens = cached_train_dataset[:500]
    
    for epoch in range(epochs):
        epoch_start = time.time()
        reldit.train()
        total_loss_epoch = 0.0
        indices = torch.randperm(len(critic_train_tokens), device=device if critic_train_tokens.is_cuda else 'cpu')
        
        for i in range(0, len(critic_train_tokens), reldit_batch_size):
            batch_indices = indices[i:i + reldit_batch_size]
            x0 = critic_train_tokens[batch_indices].to(device)
            
            t = torch.randint(1, reldit_base.num_timesteps + 1, (x0.size(0),), device=device)
            x_t, mask = reldit_base.diffusion.add_noise(x0, t)
            
            critic_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = reldit_base.transformer(x_t, t)
                probs = torch.softmax(logits.float(), dim=-1) # Upcast for Softmax stability
                critic_scores = reldit_base.get_critic_scores(probs, t)
                
            target = (~mask).float()
            # BCELoss is unsafe to autocast, so we run it in float32 outside the autocast block
            loss = bce_loss(critic_scores.float(), target)
            
            critic_scaler.scale(loss).backward()
            critic_scaler.step(critic_optimizer)
            critic_scaler.update()
            total_loss_epoch += loss.item()
            
        avg_train_loss = total_loss_epoch / max(1, len(critic_train_tokens) // reldit_batch_size)
        epoch_duration = time.time() - epoch_start
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Epoch {epoch+1}/{epochs} | Critic BCE Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.1f}s")
        
    print()
    print('Critic training complete! The CID framework is now fully integrated.')
    print('To generate highly valid graph topologies, ensure use_critic=True is passed to generate()')
