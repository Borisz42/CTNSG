# Architecture Details

The CTNSG framework is a fully decoupled generative pipeline achieving zero-hallucination and $\mathcal{O}(1)$ latency through rigorous mathematical constraints across its four synchronized modules.

## Module 1: The Macroplanner (Discrete Semantic Graph Diffusion)
Generates the logical and rhetorical blueprint as a directed acyclic discourse graph.
- **Graph Vector Tokenization (GVT):** Uses Residual Vector Quantization (RVQ) with Gumbel-Softmax reparameterization. Node ordering is canonicalized using the **Reverse Cuthill-McKee (RCM)** algorithm and paired with **Rotary Position Embeddings (RoPE)** for 99.89% exact structural reconstruction.
- **Relational Graph Diffusion Transformer (RelDiT):** Generates topology via discrete diffusion, avoiding continuous diffusion mismatch. Utilizes **Simple Iterative Denoising (SID)** and a parallel **Critic** module to prevent compounding denoising errors.
- **Directional Topology:** Preserved via **Magnetic Laplacian Positional Encodings (mLPE)** and a **Dual Attention** mechanism.
- **FandE Score Optimization:** The module is optimized against the Features and Edges (FandE) Score rather than standard Triple F1 to prevent representational saturation.

## Module 2: Semantic Prior & Supervisor (Orchestration & Retrieval)
Breaks down workloads into tractable sub-tasks and grounds the graph in verified context.
- **Arbor Task-Decoupled Planning (Arbor TDP):** Uses a Checks-and-Balances system, pairing the primary orchestrator with a Critic agent to infer parallel dependencies.
- **Probabilistic Sentential Decision Diagrams (PSDD):** Domain schemas and hard constraints are compiled offline into a mathematically guaranteed semantic prior. Projected Gradient Descent (PGD) convex optimization traverses this circuit to enforce soft constraints dynamically.
- **Hybrid RAG & Context Structuring:** Uses **Fully Autonomous Atomic Propositions (FAAP)** to enforce a "Zero Pronouns" rule. Context is pruned using **SDRT-GNN Filtering** to solve the "middle curse", and uses **Dynamic Confidence-Aware Traversal** for retrieval depth.
- **Hardware Scheduling:** The **Maestro** hierarchical scheduler predicts memory bounds to prevent Head-of-Line blocking, utilizing **Pod-Attention** to natively decouple prefill and decode phases.

## Module 3: The Realizer (High-Throughput Neuro-Symbolic Decoding)
Converts the mathematical blueprint into natural language or code using a lightweight Base LLM.
- **Virtual Node Pooling (VNPool) & LoRA:** The discrete macroplan is projected into the continuous embedding space of the Base LLM. To prevent optimization collapse during soft-prompting, the continuous vectors are stabilized by injecting trainable **LoRA adapters**.
- **Parallel Generation:** Uses **Multiple Token Prediction (MTP)** running inside a Decomposition-and-Fill parallel loop.
- **$\mathcal{O}(1)$ Grammar-Constrained Decoding:** Syntax is enforced deterministically. **Parser Stack Classification (PSC)** handles fixed schemas via $O(1)$ offline masking (up to 770$\times$ faster), while **GREATGRAMMA** handles dynamic schemas.
- **TruncProof:** Dynamically tracks the token budget and forces graceful schema closure before context limits are reached.
- **SafeLLM & OT Monitoring:** Facts are strictly extracted from FAAP via line-number extraction. A **Cross-Attention Optimal Transport (OT) Monitor** tracks middle transformer layers to detect when the LLM disengages from the source text.

## Module 4: Verification, Auditability, & Deployment
Guarantees structural safety and orchestrates multi-agent environments.
- **ATLAS Evidence Composition:** Cryptographically binds Layer-1 syntactic traces (DFA masks) with Layer-2 logical proofs (SHACL/SMT). Uses **Incremental Streaming L2 Validation** to catch logical contradictions during generation.
- **Autonomous Inspection:** L2 validators are guarded by **PaperGuard** chunk-based auditing and **Diagnostic FBR/FAR Signatures**. Code generation uses **Structured Testbench Generation (STG)**.
- **ARMOR-MAD & Brick Router:** Routes tasks via the Brick Spatial Capability Router. Debates among heterogeneous agents are resolved using Adaptive Routing for Heterogeneous Multi-Agent Debate (ARMOR-MAD) via Semantic Outlier Detection. Inter-agent communication is sanitized by **SAIGuard**.
- **TRACE Privacy Module:** Stores rules as Contributor-Stamped Functional Blocks (Fed-FBD) allowing for machine unlearning to comply with the Right to be Forgotten, protected by **Rolling-Window Histogram Audits**.
