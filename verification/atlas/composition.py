import hashlib
import json
from typing import Dict, Any

class ATLASComposer:
    """
    ATLAS Evidence Composition.
    Tracks generation provenance and creates cryptographically bound evidence 
    for the generated macro-topologies.
    """
    def __init__(self):
        self.provenance_log = []

    def log_generation_step(self, step_name: str, input_data: Any, output_data: Any) -> str:
        """
        Logs a generation step and returns a cryptographic hash representing the state.
        """
        state_str = f"{step_name}:{str(input_data)}->{str(output_data)}"
        state_hash = hashlib.sha256(state_str.encode('utf-8')).hexdigest()
        
        self.provenance_log.append({
            "step": step_name,
            "hash": state_hash
        })
        return state_hash
        
    def generate_evidence_bundle(self) -> Dict[str, Any]:
        """
        Generates the final L2 logical proof bundle.
        """
        bundle_hash = hashlib.sha256(json.dumps(self.provenance_log).encode('utf-8')).hexdigest()
        return {
            "provenance": self.provenance_log,
            "bundle_signature": bundle_hash
        }
