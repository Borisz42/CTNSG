# System Context & Directives for AI Agents

**Welcome, Agent.** 
This file provides the critical context, architectural constraints, and user goals necessary to effectively assist in developing the Canonical Tractable Neuro-Symbolic Generation (CTNSG) framework. 

Read this document carefully before making changes to the codebase.

## 🎯 The User's Primary Goal
The overarching objective is to **implement, test, and debug the AI architecture detailed in `paper.md`**. You are here to help build this fully decoupled neuro-symbolic generative pipeline from the ground up, resolving any implementation bottlenecks and mathematical challenges along the way.

## 📜 Golden Sources
- `paper.md` and `README.md` are the **Golden Sources of Truth**.
- **DO NOT MODIFY THEM.** If there is ambiguity in the codebase, refer to these files to understand the intended mathematical and structural design.

## 🏗️ Core Architectural Philosophy
The architecture do **not** attempt to solve reasoning or structural problems using standard "Prompt Engineering" or by relying on a massive LLM context window. CTNSG is a **decoupled pipeline**:
1. **Reasoning is discrete and topological.** It happens in Module 1 via Graph Diffusion (RelDiT).
2. **The LLM (Module 3) is strictly a "Realizer" (a feature extractor and syntax formatter).** It does not plan.
3. **Safety is mathematical, not probabilistic.** Hallucinations are prevented via $\mathcal{O}(1)$ grammar masking (L1) and SMT Solvers (L2), not by asking the LLM to "be careful."

### The 4-Module Structure
When debugging or implementing, isolate your thinking to the specific module:
*   **Module 1: The Macroplanner (`/macroplanner`)** - Compresses graphs via Residual Vector Quantization (RVQ) and Reverse Cuthill-McKee (RCM) ordering. Generates topologies via Discrete Diffusion (RelDiT).
*   **Module 2: Semantic Prior & Supervisor (`/orchestrator`)** - Decomposes tasks (Arbor TDP). Compiles hard constraints into Probabilistic Sentential Decision Diagrams (PSDDs). Retrieves context via FAAP (Zero Pronouns rule) and SDRT-GNN filtering.
*   **Module 3: The Realizer (`/realizer`)** - Bridges the discrete graph to the continuous LLM via Virtual Node Pooling (VNPool) and LoRA. Enforces $O(1)$ grammar masking (PSC) and prevents context-overflow via TruncProof.
*   **Module 4: Verification & Multi-Agent (`/verification`)** - Cryptographically binds L1 syntax masks to L2 SMT logical proofs (ATLAS). Handles Multi-Agent routing (ARMOR-MAD) and contagion defense (SAIGuard).

## 💻 Hardware Constraints
You must write highly optimized, memory-efficient PyTorch code.
- **Local Inference Target:** Consumer edge devices (specifically RTX 3070, 8GB VRAM). The base LLM (e.g., Phi-4-mini-instruct) consumes a significant portion, leaving only ~4-5GB for the PyTorch/Macroplanner overhead.
- **Training Target:** Kaggle Free Tier (GPU T4 x2, 16GB VRAM each). Ensure `batch_size` scaling, gradient accumulation, and memory offloading are actively managed to prevent OOM errors during GVT/RelDiT training. The scaling should be done in a way so that the training is also possible on consumer grade hardware (single GPU with 8GB VRAM).

## 🛠️ Agent Directives for Debugging & Implementation
1. **Mathematical Rigor:** The paper relies heavily on mathematical guarantees (e.g., FandE Score optimization, Magnetic Laplacian Positional Encodings, SID diffusion). Ensure your PyTorch implementations adhere strictly to these concepts rather than defaulting to standard continuous NLP techniques.
2. **Isolate the Fault (L1 vs. L2):** If a generation fails:
   - Is it malformed JSON/Code? -> Debug Module 3 ($\mathcal{O}(1)$ Grammar Masks / PSC).
   - Is it logically contradictory (e.g., $A \rightarrow B$ but $B$ happens before $A$)? -> Debug Module 4 (SMT Validator) or Module 1 (RelDiT Topology).
3. **Continuous Parameter Efficiency:** Ensure `peft` (LoRA) is strictly utilized when connecting discrete graph embeddings to the Base LLM. The LLM backbone must remain frozen to fit in memory.
4. **Use PowerShell:** The user's environment is Windows 11 with PowerShell. Do not write bash scripts or suggest Linux-only CLI commands.
