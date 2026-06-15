# Arbor Task-Decoupled Planning

This directory contains the logic for decomposing the global planning intent into decoupled sub-tasks.

## Purpose
Arbor explicitly separates the structural requirements (topology meant for the GVT and RelDiT) from the semantic requirements (textual/code data meant for the LLM Realizer). This prevents unilateral planning failures.
