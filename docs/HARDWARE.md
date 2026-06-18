# Hardware Budgeting & Constraints

The CTNSG pipeline is heavily constrained to ensure deployability on consumer edge devices while leveraging robust cloud-based environments for offline training. By offloading logical and syntactical constraints to $0$-parameter deterministic solvers (e.g., SMT solvers and PSDDs), CTNSG devotes 100% of its continuous parameter capacity to linguistic fluency.

## Local Inference Target: 8GB VRAM (RTX 3070)

The primary goal of the local inference setup is achieving $\mathcal{O}(1)$ syntax latency without Out-of-Memory (OOM) crashes.

* **Base LLM (The Realizer):** Phi-4-mini-instruct (approx. 7.6GB fp16, or 2.2GB in 4-bit quantization).
* **Total Neural Parameters:** The total active neural parameter count of the framework is approximately **3.9 Billion** (~3.8B for the Realizer + ~100M for the Macroplanner modules). This competes directly against 12B/14B unconstrained models (like Qwen-3.5 14B and Gemma-4 12B).
* **Macroplanner Margin:** Leaves ~4-5GB VRAM strictly for the Graph VQ-Transformer, RelDiT, and PyTorch overhead.
* **KV Cache & Memory Management:** Dynamic allocation managed by the **Maestro hierarchical scheduler**, which predicts memory bounds and avoids Head-of-Line blocking. **ITME (Inference Tiered Memory Expansion)** handles proactive CPU-offloading for predictable KV cache blocks, preventing VRAM exhaustion.
* **Inference Kernels:** ReSET CUDA-core small-M NVFP4 kernels for maximum autoregressive speed.
* **Context Budgeting:** **TruncProof** dynamically tracks the token budget and forces graceful schema closure before hitting the context window limit, fully preventing OOM and syntax truncation errors.

## Training Environment: Kaggle Free Tier

Training the Graph VQ-Transformer (GVT) and Relational Graph Diffusion Transformer (RelDiT) requires significant memory for intermediate activations.

* **Compute:** Kaggle P100 (16GB VRAM) or Dual T4 (2x 15GB VRAM) instances.
* **Session Lengths:** ~30 hours/week.
* **Optimization:** Checkpointing and gradient accumulation are aggressively utilized to fit batch sizes into the strict 16GB limit.
* **Environment:** `requirements.txt` driven to ensure swift notebook spin-ups.

## $\mathcal{O}(1)$ Grammar Masking Trade-offs

The **Parser Stack Classification (PSC)** mechanism enables syntax masking up to 770$\times$ faster than traditional Grammar-Constrained Decoding (GCD), isolating masking overhead from the LLM's vocabulary size to achieve unconstrained speeds.
* **Trade-off:** The primary weakness of this approach is the **offline pre-computation memory footprint**. Computing the finite-state automata for massive vocabulary models requires significant RAM and preprocessing time before the $\mathcal{O}(1)$ latency can be achieved during inference.
