# Hybrid RAG & FAAP Context

This directory manages knowledge retrieval and context structuring for the Supervisor module.

## Purpose
It transforms raw corpus text into a mathematically structured format that guarantees coreference safety and prevents the LLM "middle curse" before context is ingested.

## Key Components

### 1. Fully Autonomous Atomic Propositions (FAAP)
Raw corpus text is decomposed into autonomous facts that enforce a strict **Zero Pronouns rule**. This ensures that all retrieved context chunks are completely independent and co-reference safe, preventing the Realizer from misattributing actions to the wrong entities.

### 2. SDRT-GNN Context Filtering
To solve the "middle curse" in long-context parsing, the text is converted into a Segmented Discourse Representation Theory (SDRT) graph. A Graph Neural Network (GNN) node classifier evaluates this graph to actively prune non-salient utterances before they ever reach the LLM's context window.

### 3. Dynamic Confidence-Aware Traversal
Hybrid RAG retrieval uses a dense + sparse embedding index. Instead of using static hop limits (e.g., retrieve top-K), it utilizes **Dynamic Confidence-Aware Traversal**. This allows an LLM-as-a-judge to determine exactly how deep to explore the graph on a per-node basis.
