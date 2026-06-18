import os
import gc
import time
import re

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    import sys
    print("Installing datasets library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

try:
    from llama_cpp import Llama
except ImportError:
    import subprocess
    import sys
    print("Installing llama-cpp-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
    from llama_cpp import Llama

try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, HfApi

NUM_EVALS = 5  # Keep subset to N=5

def download_gguf(repo_id):
    print(f"  -> Locating Q4_K_M GGUF in repo: {repo_id}...")
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id)
        # Try to find a q4_k_m or similar 4-bit quantization
        target_file = None
        for f in files:
            if "q4_k_m" in f.lower() and f.endswith(".gguf"):
                target_file = f
                break
        if not target_file:
            for f in files:
                if "q4_0" in f.lower() and f.endswith(".gguf"):
                    target_file = f
                    break
        if not target_file:
            for f in files:
                if f.endswith(".gguf"):
                    target_file = f
                    break
                    
        if not target_file:
            raise FileNotFoundError(f"No .gguf file found in {repo_id}")
            
        print(f"  -> Downloading {target_file} from {repo_id}...")
        path = hf_hub_download(repo_id=repo_id, filename=target_file)
        return path
    except Exception as e:
        print(f"  -> Error fetching GGUF from {repo_id}: {e}")
        return None

def build_prompt(sample, context_text):
    prompt = (
        f"Context: {context_text}\n\n"
        f"Question: {sample['question']}\n"
        f"A) {sample['choice_A']}\n"
        f"B) {sample['choice_B']}\n"
        f"C) {sample['choice_C']}\n"
        f"D) {sample['choice_D']}\n"
        f"Please provide the correct option letter (A, B, C, or D).\nAnswer:"
    )
    return prompt

def evaluate_baseline(repo_id, dataset_subset):
    print(f"\n{'='*60}")
    print(f"Testing Untruncated Baseline (GGUF): {repo_id}")
    print(f"Constraints: 4-bit GGUF, n_ctx=65536, RAM Spillover Enabled")
    print(f"{'='*60}")
    
    correct = 0
    total = len(dataset_subset)
    
    model_path = download_gguf(repo_id)
    if not model_path:
        return 0.0
        
    try:
        print(f"  -> Loading Llama.cpp engine (n_ctx=65536, n_gpu_layers=-1)...")
        # Initialize with massive context window. VRAM overflows will safely spill to system RAM.
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=65536,
            verbose=False
        )
        
        for i, sample in enumerate(dataset_subset):
            print(f"  -> Testing example {i+1}/{total} (Domain: {sample['domain']})")
            
            # Pass the FULL, untruncated context to the model!
            prompt = build_prompt(sample, sample['context'])
            
            start_time = time.time()
            output = llm(
                prompt,
                max_tokens=10,
                stop=["\n", "Question:"],
                echo=False
            )
            end_time = time.time()
            
            response = output['choices'][0]['text'].strip()
            
            pred = "Unknown"
            match = re.search(r'\b([A-D])\b', response)
            if match:
                pred = match.group(1)
            elif response.startswith("A") or response.startswith("B") or response.startswith("C") or response.startswith("D"):
                pred = response[0]
                
            if pred == sample['answer']:
                correct += 1
                
            print(f"     [Result] Groundtruth: {sample['answer']}, Prediction: {pred} (Time: {end_time - start_time:.1f}s)")
        
        del llm
        gc.collect()
        
        score = (correct / total) * 100
        print(f"  [FINAL] Baseline {repo_id} scored: {score:.1f}%")
        return score
            
    except Exception as e:
        print(f"  [RESULT] ERROR: Baseline failed with exception: {e}")
        if 'llm' in locals(): del llm
        gc.collect()
        return 0.0


def main():
    print("Loading HuggingFace THUDM/LongBench-v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    
    subset = [dataset[i] for i in range(min(NUM_EVALS, len(dataset)))]
    print(f"Loaded {len(subset)} examples for evaluation.\n")
    
    scores = {}
    
    # 1. Qwen (Untruncated GGUF)
    scores["Qwen3.5-4B"] = evaluate_baseline("unsloth/Qwen3.5-4B-GGUF", subset)
    
    # 2. Gemma (Untruncated GGUF)
    scores["gemma-4-E4B"] = evaluate_baseline("unsloth/gemma-4-E4B-it-GGUF", subset)
    
    # 3. Phi (Untruncated GGUF)
    scores["Phi-4-mini-instruct"] = evaluate_baseline("unsloth/Phi-4-mini-instruct-GGUF", subset)

    print("\n" + "="*60)
    print("FINAL UNTRUNCATED EVAL SCORES (LongBench v2 - N=5)")
    print("="*60)
    for model, score in scores.items():
        print(f"{model:<30}: {score:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()
