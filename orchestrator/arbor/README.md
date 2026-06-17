# Arbor Task-Decoupled Planning

This directory contains the logic for decomposing the global planning intent into computationally tractable, decoupled sub-tasks.

## Purpose
Arbor explicitly separates the structural requirements (topology meant for the GVT and RelDiT) from the semantic requirements (textual/code data meant for the LLM Realizer). This strict separation prevents unilateral planning failures common in standard autoregressive models.

## Key Components

### 1. Arbor Checks-and-Balances System
The global task is decomposed into a strict Directed Acyclic Graph (DAG) of sub-tasks. To prevent single points of failure in planning, the primary orchestrator is paired with a **Critic agent**. This dual-agent setup actively evaluates the decomposition strategy.

### 2. True Parallel DAG Inference
Instead of naively sorting linear agent traces (e.g., from ToolBench), Arbor actively infers parallel dependencies. Independent execution steps are identified and parallelized before Reverse Cuthill-McKee (RCM) canonicalization, enabling true parallel DAG generation rather than linear sequence execution.
