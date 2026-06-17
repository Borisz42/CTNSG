# L2 Validation (SHACL & SMT)

This directory contains the formal logic validators for the CTNSG pipeline.

## Purpose
While Layer-1 (L1) parsing masks provide $\mathcal{O}(1)$ syntax safety (e.g., ensuring valid JSON structure), L1 cannot guarantee semantic correctness (e.g., ensuring "start_time < end_time"). This directory implements Layer-2 (L2) validation. It uses `z3-solver` and `pyshacl` to formally verify the generated macro-topologies against predefined semantic rules and constraints, ensuring no logical contradictions exist before the Realizer processes the graph.

## Key Mechanisms

### 1. The "Syntax vs. Semantics" Gap
L1 constraints handle the grammar, and L2 catches semantic bypasses. This module forms the backbone of the **Audit-Guided Repair (AGR)** loop. If an L2 check fails (e.g., a constraint is violated), the SMT solver mathematically identifies the contradiction and passes the error trace back to the RelDiT diffusion module to resample the specific failing nodes.

### 2. Formal Tooling
- **Z3 Theorem Prover (`z3-solver`):** Used to solve integer, boolean, and real-valued algebraic constraints derived from the graph topology.
- **SHACL (`pyshacl`):** Used to validate graph shapes and RDF-style entity relations against strict ontologies.
