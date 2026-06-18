import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train_arbor_sft(
    model_name: str,
    arbor_graphs,
    device: str = "cuda",
    epochs: int = 3,
    lr: float = 2e-5,
    r: int = 8,
    alpha: int = 16,
    max_length: int = 512,
    batch_size: int = None,
    output_dir: str = "ctnsg_export"
):
    """
    Fine-tunes the base LLM with LoRA on Arbor true DAGs to perform agentic task decomposition.
    """
    if batch_size is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Phi-4-mini is 3.8B parameters. We assume ~5GB base overhead for model weights and optimizer states,
            # and ~2.5GB per batch item for activations with max_length=512.
            # device_map="auto" splits layers across GPUs (pipeline parallel), so the full batch passes through each GPU.
            # Thus, we do NOT multiply the calculated physical batch size by num_gpus.
            bs_estimate = max(1, int((vram_gb - 5.0) / 2.5))
            powers_of_2 = [1, 2, 4, 8, 16, 32]
            batch_size = max([p for p in powers_of_2 if p <= bs_estimate], default=1)
            print(f"Dynamically set batch size to {batch_size} (GPUs: {num_gpus}, VRAM/GPU: {vram_gb:.1f} GB)")
        else:
            batch_size = 1
            print("CUDA not available. Defaulting batch size to 1.")
    print("Loading Base LLM for Arbor SFT...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_llm = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    base_llm = prepare_model_for_kbit_training(base_llm)
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    arbor_model = get_peft_model(base_llm, lora_config)
    arbor_model.print_trainable_parameters()
    
    from transformers import get_linear_schedule_with_warmup
    import json
    
    print("Formatting Prompt-Completion Pairs and sorting by length (Dynamic Batching)...")
    optimizer = torch.optim.AdamW(arbor_model.parameters(), lr=lr)
    
    limit = min(5000, len(arbor_graphs))
    dataset_texts = []
    for idx in range(limit):
        graph = arbor_graphs[idx]
        goal = graph.get("goal", f"Task {idx}")
        prompt = f"<|im_start|>system\nYou are the Arbor Supervisor. Decompose this task into a JSON DAG.<|im_end|>\n<|im_start|>user\n{goal}<|im_end|>\n<|im_start|>assistant\n"
        node_names = graph.get("node_names", [])
        text_edges_raw = graph.get("text_edges", [])
        text_edges = []
        for edge in text_edges_raw:
            if len(edge) == 2:
                text_edges.append({
                    "source": node_names[edge[0]],
                    "target": node_names[edge[1]],
                    "relation": "depends_on"
                })
        completion = json.dumps({"nodes": node_names, "edges": text_edges})
        text = prompt + completion + "<|im_end|>"
        dataset_texts.append(text)
        
    # Sort by length to minimize padding within batches
    dataset_texts.sort(key=len)
    
    num_batches = (limit + batch_size - 1) // batch_size
    num_training_steps = epochs * num_batches
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=min(100, int(0.1 * num_training_steps)), 
        num_training_steps=num_training_steps
    )
    
    accumulation_steps = max(1, 16 // batch_size)
    print(f"Targeting Effective Batch Size 16. Physical BS: {batch_size}, Accumulation Steps: {accumulation_steps}")
    
    arbor_model.train()
    from tqdm import tqdm
    for epoch in range(epochs):
        total_loss = 0
        print(f"\nStarting Epoch {epoch+1}/{epochs}...")
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        for b_idx in pbar:
            batch_texts = dataset_texts[b_idx * batch_size : (b_idx + 1) * batch_size]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            
            labels = inputs["input_ids"].clone()
            # Mask out padding tokens from labels so they don't contribute to loss
            labels[inputs["attention_mask"] == 0] = -100
            
            outputs = arbor_model(**inputs, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (b_idx + 1) % accumulation_steps == 0 or (b_idx + 1) == num_batches:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            curr_loss = loss.item() * accumulation_steps
            total_loss += curr_loss
            pbar.set_postfix({"loss": f"{curr_loss:.4f}"})
            
        print(f"Epoch {epoch+1} Completed | Arbor SFT Loss: {total_loss/num_batches:.4f}")
        
    print("Saving Arbor LoRA weights...")
    os.makedirs(output_dir, exist_ok=True)
    arbor_model.save_pretrained(os.path.join(output_dir, "arbor_lora_weights"))
    
    # Save the 384->256 projection layer export (simulated retrieval from preprocessing)
    proj_layer = torch.nn.Linear(384, 256)
    torch.save(proj_layer.state_dict(), os.path.join(output_dir, "encoder_projection_weights.pt"))
    
    return arbor_model, tokenizer, proj_layer
