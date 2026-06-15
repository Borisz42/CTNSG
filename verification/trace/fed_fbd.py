from typing import Dict, Any, List

class TRACEFedFBD:
    """
    TRACE Module with Contributor-Stamped Functional Blocks (Fed-FBD).
    Allows surgical machine unlearning without recompiling the entire PSDD.
    """
    def __init__(self):
        # Maps contributor_id -> List of functional blocks (rules/constraints)
        self.registry: Dict[str, List[Any]] = {}
        
    def register_block(self, contributor_id: str, rule_block: Any):
        """ Registers a new functional constraint stamped with the contributor's ID. """
        if contributor_id not in self.registry:
            self.registry[contributor_id] = []
        self.registry[contributor_id].append(rule_block)
        
    def unlearn_contributor(self, contributor_id: str) -> bool:
        """
        Surgically removes all rules associated with a specific contributor 
        (Right to be Forgotten).
        """
        if contributor_id in self.registry:
            del self.registry[contributor_id]
            print(f"[TRACE] Surgically unlearned contributor: {contributor_id}")
            return True
        return False
        
    def get_active_constraints(self) -> List[Any]:
        """ Retrieves all currently active rules for dynamic compilation. """
        active = []
        for blocks in self.registry.values():
            active.extend(blocks)
        return active

class RollingWindowAudit:
    """
    Protects the replay buffer from "Amnesia" attacks via Rolling-Window Histogram Audits.
    Detects catastrophic forgetting of historical rules.
    """
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.15):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.history_histogram = {}
        self.current_window = []
        
    def record_activation(self, rule_type: str):
        self.current_window.append(rule_type)
        if len(self.current_window) > self.window_size:
            # Pop oldest
            self.current_window.pop(0)
            
    def audit(self) -> bool:
        """
        Compares the current window histogram against the historical baseline to detect 
        if certain rules have been suspiciously "forgotten" (amnesia attack).
        """
        if len(self.current_window) < self.window_size:
            return True # Not enough data yet
            
        # Calculate current histogram
        current_hist = {}
        for r in self.current_window:
            current_hist[r] = current_hist.get(r, 0) + 1
            
        # For a true implementation, we compare current_hist frequencies to historical baselines
        # Here we mock the check
        for rule, count in current_hist.items():
            freq = count / self.window_size
            baseline = self.history_histogram.get(rule, 0.0)
            if baseline > 0 and (baseline - freq) > self.drift_threshold:
                print(f"[Amnesia Alert] Rule {rule} frequency dropped by {baseline - freq:.2f}")
                return False
                
        # Update baseline via moving average
        for rule, count in current_hist.items():
            freq = count / self.window_size
            self.history_histogram[rule] = 0.9 * self.history_histogram.get(rule, freq) + 0.1 * freq
            
        return True
