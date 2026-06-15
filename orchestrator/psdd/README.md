# PSDD Semantic Prior & Supervisor

This directory contains the **Probabilistic Sentential Decision Diagrams (PSDD)** implementation.

## Purpose
The PSDD acts as a tractable semantic prior. It strictly limits the generated token budget before Realizer generation using the **TruncProof** convex optimizer. By calculating dynamic boundaries based on the LLM's context window and the required schema closure tokens, it eliminates the risk of context overflow and "semantic collapse".
