# Hardware Budgeting & Constraints

The CTNSG pipeline is heavily constrained to ensure deployability on consumer edge devices while leveraging robust cloud-based environments for offline training.

## Local Inference Target: 8GB VRAM (RTX 3070)

The primary goal of the local inference setup is achieving $\mathcal{O}(1)$ syntax latency without Out-of-Memory (OOM) crashes.

* **Syntax Model:** Qwen-2.5-3B-Instruct (approx. 6GB fp16, or 2.5GB in 4-bit quantization).
* **Macroplanner Margin:** Leaves ~4-5GB VRAM strictly for the Graph VQ-Transformer, RelDiT, and PyTorch overhead.
* **KV Cache:** Dynamic allocation managed by the **Maestro hierarchical scheduler**, which predicts memory bounds and avoids Head-of-Line blocking.
* **Inference Kernels:** ReSET CUDA-core small-M NVFP4 kernels for maximum autoregressive speed.

## Training Environment: Kaggle Free Tier

Training the Graph VQ-Transformer (GVT) and Relational Graph Diffusion Transformer (RelDiT) requires significant memory for intermediate activations.

* **Compute:** Kaggle P100 (16GB VRAM) or Dual T4 (2x 15GB VRAM) instances.
* **Session Lengths:** ~30 hours/week.
* **Optimization:** Checkpointing and gradient accumulation are aggressively utilized to fit batch sizes into the 16GB limit.
* **Environment:** `requirements.txt` driven to ensure swift notebook spin-ups.
