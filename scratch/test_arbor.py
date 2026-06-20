import sys
import os
sys.path.append(os.path.abspath('..'))

import json
from orchestrator.arbor.planner import ArborPlanner

# We can simulate the output if we load the model, but maybe we can just look at the logs.
# Let's write a dummy response and test parsing
planner = ArborPlanner() # No LLM

skeleton_edges = [
    {"source": "n_47", "target": "n_20", "relation": "depends_on"}
]

subtasks = planner.generate_subtask_dag("Test", skeleton_edges=skeleton_edges, num_skeleton_nodes=2)
print("Subtasks:", subtasks)
