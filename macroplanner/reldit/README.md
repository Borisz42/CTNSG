# Module 2: Relational Diffusion Transformer (RelDiT)

This module implements the generative engine of the Macroplanner. Given the discrete token sequences produced by the GVT (Module 1), RelDiT utilizes a non-autoregressive discrete diffusion process to generate novel graph topologies from pure noise.

## Key Components

### 1. `diffusion.py` (Absorbing Discrete Diffusion)
Unlike continuous Gaussian diffusion used in image models, graph tokens are discrete. We implement an **Absorbing State Diffusion Schedule**. 
- During the forward (noising) process, tokens are progressively overwritten by a special `[MASK]` token.
- During the reverse (denoising) process, the model learns to predict the true categorical distribution of the masked tokens conditioned on the unmasked context.

### 2. `transformer.py` (Bidirectional Transformer Encoder)
Because graph tokens ordered by Reverse Cuthill-McKee (RCM) have dependencies in both directions, we use a Bidirectional `TransformerEncoder`. 
- It acts as the noise-prediction backbone, attending to the entire corrupted sequence at once to predict the unmasked categorical logits for the RVQ arrays.
- It bypasses the causal masking bottleneck inherent in autoregressive LLMs, accelerating parallel graph generation.

### 3. `model.py` (RelDiT Generator with SID & Critic)
The orchestrator script that ties the schedule and the transformer together. 
- **Training Mode:** Applies random masking according to the diffusion schedule and computes the categorical Cross-Entropy loss against the true GVT tokens.
- **Generation Mode (Iterative Unmasking):** Starts with a fully masked sequence (`[MASK, MASK, MASK...]`) and iteratively unmasks the most confident token predictions step-by-step until a complete, valid topological token sequence is synthesized.
- **Critic & SID:** Incorporates a parallel **Critic module** to actively evaluate intermediate states and **Simple Iterative Denoising (SID)** to re-corrupt low-likelihood topological elements during generation, preventing compounding errors.
