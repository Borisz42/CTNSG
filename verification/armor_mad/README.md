# ARMOR-MAD Orchestrator Routing

This directory contains the Adaptive Routing for Heterogeneous Multi-Agent Debate (ARMOR-MAD) logic.

## Purpose
It orchestrates the full end-to-end dataflow between specialized agents, routing queries efficiently and resolving debate conflicts among heterogeneous frontier models.

## Key Components

### 1. Brick Spatial Capability Router
Model queries are dynamically routed via the **Brick Spatial Capability Router**, which evaluates tasks against six capability dimensions and dispatches them based on geometric cost-penalty rules. This ensures specialized reasoning is handled by the appropriate agent type.

### 2. Semantic Outlier Detection (SOD)
To resolve distinct reasoning styles and conflicts during multi-agent debate, ARMOR-MAD utilizes Semantic Outlier Detection. This mathematically identifies and isolates diverging reasoning paths before a consensus failure occurs.

### 3. SAIGuard Contagion Defense
Inter-agent communication is proactively sanitized by **SAIGuard (Simulation-aware Interception Guard)**. This module monitors state-reconstruction anomalies during message passing, intercepting toxic or hallucinated strings to prevent hallucination contagion across the Multi-Agent System (MAS) network.
