# ATLAS Evidence Composition

This directory tracks generation provenance and enforces continuous validation and auditability.

## Purpose
It cryptographically binds the Layer-1 syntactic generation traces with the Layer-2 logical proofs into verifiable evidence bundles. This ensures full auditability of the generated output, allowing any downstream system to independently verify that the output adheres to the specified constraints.

## Key Components

### 1. Incremental Streaming L2 Validation
Instead of waiting for generation to complete, ATLAS passes partial generation graphs to SMT solvers in real-time. This **Incremental Streaming L2 Validation** catches logical contradictions immediately during the generation phase, enabling faster aborts and preventing the Realizer from wasting compute on doomed branches.

### 2. PaperGuard Auditing
L2 validators are guarded by **PaperGuard** chunk-based auditing. This prevents adversarial repackaging and detects "accidental cancellation" bias in the LLM-as-a-judge mechanisms.

### 3. Structured Testbench Generation (STG)
For code-generation workflows, ATLAS completely replaces LLM-written tests. It utilizes **Structured Testbench Generation (STG)**, generating fully deterministic testing suites derived directly from the verified graph topology, guaranteeing that tests perfectly cover the required logic boundaries.
