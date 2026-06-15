# L2 Validation (SHACL & SMT)

This directory contains the formal logic validators.

## Purpose
It uses `z3-solver` and `pyshacl` to formally verify the generated macro-topologies against predefined semantic rules and constraints. It ensures that the generated graph structure contains no logical contradictions or schema violations before the LLM Realizer processes it.
