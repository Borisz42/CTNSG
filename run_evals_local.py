import os
import json
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

# Top-level definition for Windows multiprocessing compatibility
def execute_code(code, test_case, queue):
    """
    Restricted execution environment for HumanEval.
    Must be at the top level so Windows 'spawn' can pickle it.
    """
    try:
        exec_globals = {"__builtins__": {}}
        exec_locals = {}
        # Execute the generated function alongside the hidden test cases
        exec(code + "\n" + test_case, exec_globals, exec_locals)
        queue.put(True)
    except Exception:
        queue.put(False)


def main():
    parser = argparse.ArgumentParser(description="Run CTNSG Benchmarks Locally")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples per benchmark (set to 0 for exhaustive)")
    args = parser.parse_args()
    
    num_samples = args.samples if args.samples > 0 else None

    # Late imports to speed up CLI help and avoid issues before multiprocessing initialization
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library: pip install datasets")
        return

    from orchestrator.arbor.planner import ArborPlanner
    from realizer.realizer import CTNSGRealizer
    from contracts.graph_schema import DiscourseGraph, SemanticNode, SemanticEdge

    print("\n========================================================")
    print("   CTNSG Local Evaluation Harness (Windows Optimized)   ")
    print("========================================================")
    
    sample_text = f"{num_samples} samples" if num_samples else "EXHAUSTIVE (All samples)"
    print(f"\nInitializing CTNSGRealizer (Model: Phi-4-mini-instruct, 4-bit, Target VRAM: ~2.5GB)...")
    realizer = CTNSGRealizer(model_name="unsloth/Phi-4-mini-instruct")
    
    # -------------------------------------------------------------------------
    # 1. GSM8K Evaluation (JSON Schema Masking)
    # -------------------------------------------------------------------------
    print(f"\n[1/3] Evaluating GSM8K (Math & Logic) on {sample_text}...")
    gsm8k = load_dataset('gsm8k', 'main', split='test')
    if num_samples: gsm8k = gsm8k.select(range(num_samples))

    gsm8k_correct = 0
    gsm8k_schema = {
        "type": "object", 
        "properties": {
            "reason_chain": {"type": "string"}, 
            "final_answer": {"type": "number"}
        }
    }
    
    for item in gsm8k:
        mock_graph = DiscourseGraph(graph_id="gsm", nodes=[SemanticNode("n1", "math", 0)], edges=[])
        res = realizer.generate(mock_graph, gsm8k_schema, context_lines=0, prompt=item['question'])
        try:
            ans = float(json.loads(res['text'])['final_answer'])
            gt = float(item['answer'].split('####')[1].strip())
            if abs(ans - gt) < 1e-4: 
                gsm8k_correct += 1
        except Exception:
            pass
            
    ctnsg_gsm8k = (gsm8k_correct / len(gsm8k)) * 100
    print(f"-> CTNSG GSM8K Score: {ctnsg_gsm8k:.1f}%")

    # -------------------------------------------------------------------------
    # 2. MMLU-Pro Evaluation (Character Schema Masking)
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Evaluating MMLU-Pro (Reasoning) on {sample_text}...")
    mmlu = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
    if num_samples: mmlu = mmlu.select(range(num_samples))

    mmlu_correct = 0
    mmlu_schema = {"type": "string", "pattern": "^[A-J]$"}
    
    for item in mmlu:
        prompt = item['question'] + "\nOptions:\n"
        for i, opt in enumerate(item['options']): 
            prompt += f"{chr(65+i)}: {opt}\n"
            
        mock_graph = DiscourseGraph(graph_id="mmlu", nodes=[SemanticNode("n1", "qa", 0)], edges=[])
        res = realizer.generate(mock_graph, mmlu_schema, context_lines=0, prompt=prompt)
        
        if res['text'].strip() == item['answer']: 
            mmlu_correct += 1
            
    ctnsg_mmlu = (mmlu_correct / len(mmlu)) * 100
    print(f"-> CTNSG MMLU-Pro Score: {ctnsg_mmlu:.1f}%")

    # -------------------------------------------------------------------------
    # 3. HumanEval Evaluation (Restricted Execution)
    # -------------------------------------------------------------------------
    print(f"\n[3/3] Evaluating HumanEval (Coding) on {sample_text}...")
    heval = load_dataset('openai_humaneval', split='test')
    if num_samples: heval = heval.select(range(num_samples))

    heval_correct = 0
    for item in heval:
        mock_graph = DiscourseGraph(graph_id="heval", nodes=[SemanticNode("n1", "code", 0)], edges=[])
        res = realizer.generate(mock_graph, {}, context_lines=0, prompt=item['prompt'])
        full_code = item['prompt'] + res['text']
        
        queue = multiprocessing.Queue()
        # execution worker spawned using the top-level execute_code function
        p = multiprocessing.Process(target=execute_code, args=(full_code, item['test'], queue))
        p.start()
        p.join(timeout=3.0)
        
        success = False
        if p.is_alive():
            p.terminate()
            p.join()
        else:
            try: 
                success = queue.get_nowait()
            except: 
                pass
        
        if success: 
            heval_correct += 1
            
    ctnsg_heval = (heval_correct / len(heval)) * 100
    print(f"-> CTNSG HumanEval Score: {ctnsg_heval:.1f}%")

    # -------------------------------------------------------------------------
    # 4. Render Matplotlib Graph
    # -------------------------------------------------------------------------
    print("\n--- Rendering Benchmark Comparisons ---")
    benchmarks = ['MMLU-Pro', 'GSM8K', 'HumanEval']
    qwen_scores = [79.1, 89.5, 73.0]
    gemma_scores = [69.4, 89.2, 52.0]
    phi_scores = [52.8, 88.6, 74.4]
    ctnsg_scores = [ctnsg_mmlu, ctnsg_gsm8k, ctnsg_heval]

    x = np.arange(len(benchmarks))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 1.5*width, qwen_scores, width, label='Qwen-3.5-4B', color='#d62728')
    rects2 = ax.bar(x - 0.5*width, gemma_scores, width, label='Gemma 4 E4B', color='#ff7f0e')
    rects3 = ax.bar(x + 0.5*width, phi_scores, width, label='Phi-4-mini', color='#2ca02c')
    rects4 = ax.bar(x + 1.5*width, ctnsg_scores, width, label='CTNSG (~3.9B)', color='#1f77b4')

    ax.set_ylabel('Accuracy / Pass Rate (%)')
    ax.set_title('CTNSG Framework vs Autoregressive Baselines (4B Class)')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    plt.show()

    print("\nLocal evaluation suite finished successfully!")


if __name__ == '__main__':
    # freeze_support is necessary for Windows multiprocessing executable stability
    multiprocessing.freeze_support()
    main()
