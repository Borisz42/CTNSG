import hashlib
import json
from typing import Dict, Any, Tuple

class PaperGuard:
    """
    PaperGuard chunk-based auditing.
    Detects adversarial repackaging by cryptographically hashing chunks of the L2 validator proofs.
    """
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.audit_ledger = {}

    def _hash_chunk(self, chunk: str) -> str:
        return hashlib.sha256(chunk.encode('utf-8')).hexdigest()

    def audit_proof(self, proof_id: str, l2_proof_content: str) -> bool:
        """
        Splits the proof into chunks, hashes them, and checks against the ledger 
        to detect if the proof has been adversarially repackaged/tampered with.
        """
        chunks = [l2_proof_content[i:i+self.chunk_size] for i in range(0, len(l2_proof_content), self.chunk_size)]
        chunk_hashes = [self._hash_chunk(c) for c in chunks]
        
        if proof_id in self.audit_ledger:
            # Verify against existing ledger entry
            stored_hashes = self.audit_ledger[proof_id]
            if stored_hashes != chunk_hashes:
                print(f"[PaperGuard Alert] Proof {proof_id} has been tampered with or repackaged!")
                return False
            return True
        else:
            # Register new proof
            self.audit_ledger[proof_id] = chunk_hashes
            return True

class DiagnosticSignature:
    """
    Diagnostic FBR (False Bias Rate) and FAR (False Acceptance Rate) Signatures.
    Prevents "accidental cancellation" bias in the LLM-as-a-judge by calibrating decisions 
    against known baselines.
    """
    def __init__(self, fbr_threshold: float = 0.05, far_threshold: float = 0.05):
        self.fbr_threshold = fbr_threshold
        self.far_threshold = far_threshold
        
    def validate_judge_decision(self, judge_confidence: float, historical_bias_offset: float) -> Tuple[bool, str]:
        """
        Adjusts the judge's confidence using the bias offset. 
        If the adjusted confidence crosses the threshold, it triggers a diagnostic warning.
        """
        calibrated_confidence = judge_confidence - historical_bias_offset
        
        if calibrated_confidence < self.fbr_threshold:
            return False, "FBR Violation: Judge is excessively canceling valid outputs."
        if calibrated_confidence > (1.0 - self.far_threshold):
            return False, "FAR Violation: Judge is excessively accepting invalid outputs."
            
        return True, "Judge decision within diagnostic tolerance."

def test_paperguard():
    pg = PaperGuard()
    proof_content = "smt_solver_output: sat\n" * 100
    
    # Register
    pg.audit_proof("proof_001", proof_content)
    
    # Verify exact
    is_valid = pg.audit_proof("proof_001", proof_content)
    assert is_valid == True
    
    # Verify tampered
    tampered_content = proof_content + "adversarial_injection: true"
    is_valid = pg.audit_proof("proof_001", tampered_content)
    assert is_valid == False
    
    print("PaperGuard tests passed.")

if __name__ == "__main__":
    test_paperguard()
