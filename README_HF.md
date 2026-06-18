---
license: cc-by-sa-4.0
task_categories:
- text-generation
- graph-ml
language:
- en
---

# CTNSG Graph Curriculum Dataset

This dataset contains preprocessed graphs from **WebNLG (v3.0)**, **ATOMIC**, and **Spider**. 
It is explicitly designed for the Canonical Tractable Neuro-Symbolic Generation (CTNSG) framework.

## Preprocessing
All raw data has been parsed into continuous node and edge embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
Crucially, the graphs have been mathematically canonicalized using the **Reverse Cuthill-McKee (RCM)** algorithm. 
This minimizes the adjacency-matrix bandwidth, ensuring that structurally proximal nodes have nearby indices and breaking node symmetry reproducibly.

### Topological Logic
*   **ATOMIC** events are processed using an Entity Re-use Mechanism to create multi-hop Directed Acyclic Graphs (DAGs) representing deep causal chains.
*   **Spider** text-to-SQL logic queries are structured into schema DAGs (Tables -> Columns).
*   **Arbor TDP:** Linear agent traces are evaluated to find independent steps and mapped into true parallel DAGs.
*   **SDRT:** Discourse parsing maps Elementary Discourse Units (EDUs) connected by rhetorical relations.
*   **Verification:** SAIGuard multi-agent contagion graphs and Brick Router chat transcripts with Semantic Outlier Detection (SOD) scoring.

## Splits
- WebNLG (v3.0): 13,211 train
- ATOMIC: 202,271 train (Causal Reasoning)
- Spider: 7,000 train (SQL Generation)
- FAAP (Fully Autonomous Atomic Propositions): 6,812 train (Decontextualization)

## Privacy & Legal
This dataset complies with the Right to be Forgotten via the CTNSG TRACE module architecture. 
WebNLG and Spider subsets are provided under CC BY-SA 4.0.
