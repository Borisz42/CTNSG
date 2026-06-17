# PSDD Semantic Prior & Supervisor

This directory contains the **Probabilistic Sentential Decision Diagrams (PSDD)** implementation.

## Purpose
The PSDD acts as a mathematically guaranteed semantic prior. By compiling domain schemas and hard logic entirely offline, it renders logical hallucinations mathematically impossible during inference.

## Key Mechanisms

### 1. Hard Constraints (Offline Compilation)
Domain schemas, rules, and hard logical bounds are compiled offline into the static PSDD sub-circuits. During inference, the Supervisor chains these sub-circuits together, providing the LLM with a safe envelope of valid generation paths.

### 2. Soft Constraints (PGD Convex Optimization)
While hard rules are static, soft constraints (e.g., tone, fairness, attribute coverage) are dynamic. These are enforced via **Projected Gradient Descent (PGD)** over the convex hull of the PSDD. This allows the model to optimize continuous soft variables without ever violating the discrete hard logic boundaries.

### 3. TruncProof Context Bounding
The PSDD strictly limits the generated token budget before Realizer generation using the **TruncProof** module. By calculating dynamic boundaries based on the LLM's context window and the required schema closure tokens, it forces graceful schema closure and completely eliminates the risk of context overflow or "semantic collapse".
