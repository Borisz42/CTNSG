import os
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
    output_dir: str = "ctnsg_export"
):
    """
    Fine-tunes the base LLM with LoRA on Arbor true DAGs to perform agentic task decomposition.
    """
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
    print("Formatting Prompt-Completion Pairs & Training...")
    optimizer = torch.optim.AdamW(arbor_model.parameters(), lr=lr)
    
    limit = min(5000, len(arbor_graphs))
    num_training_steps = epochs * limit
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=min(100, int(0.1 * num_training_steps)), 
        num_training_steps=num_training_steps
    )
    
    arbor_model.train()
    from tqdm import tqdm
    for epoch in range(epochs):
        total_loss = 0
        print(f"\nStarting Epoch {epoch+1}/{epochs}...")
        pbar = tqdm(range(limit), desc=f"Epoch {epoch+1}")
        for idx in pbar:
            graph = arbor_graphs[idx]
            goal = graph.get("goal", f"Task {idx}")
            
            prompt = f"<|im_start|>system\nYou are the Arbor Supervisor. Decompose this task into a JSON DAG.<|im_end|>\n<|im_start|>user\n{goal}<|im_end|>\n<|im_start|>assistant\n"
            
            node_names = graph.get("node_names", [])
            text_edges = graph.get("text_edges", [])
            
            import json
            completion = json.dumps({"nodes": node_names, "edges": text_edges})
            text = prompt + completion + "<|im_end|>"
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            
            outputs = arbor_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            curr_loss = loss.item()
            total_loss += curr_loss
            pbar.set_postfix({"loss": f"{curr_loss:.4f}"})
            
        print(f"Epoch {epoch+1} Completed | Arbor SFT Loss: {total_loss/limit:.4f}")
        
    print("Saving Arbor LoRA weights...")
    os.makedirs(output_dir, exist_ok=True)
    arbor_model.save_pretrained(os.path.join(output_dir, "arbor_lora_weights"))
    
    # Save the 384->256 projection layer export (simulated retrieval from preprocessing)
    proj_layer = torch.nn.Linear(384, 256)
    torch.save(proj_layer.state_dict(), os.path.join(output_dir, "encoder_projection_weights.pt"))
    
    return arbor_model, tokenizer, proj_layer
