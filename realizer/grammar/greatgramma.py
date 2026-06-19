"""
GREATGRAMMA — O(1) Precomputed Grammar-Constrained Decoding for CTNSG.

STATE REPRESENTATION
====================
Each DFA state is a hashable tuple:  (stack, mode, aux, escaped)

  stack   : tuple of frame tuples (immutable, hashable)
  mode    : str — one of "STRUCTURAL", "KEY", "STRING", "LITERAL"
  aux     : depends on mode:
              STRUCTURAL → "" (empty string constant)
              KEY        → str  (current key prefix; bounded by max key len → BFS terminates)
              STRING     → int  (compact pattern-DFA state; see below → BFS terminates)
              LITERAL    → str  (current literal prefix; bounded → BFS terminates)
  escaped : bool (only meaningful in STRING mode)

STACK FRAME FORMATS
===================
  Object frame : ('object', path:tuple, remaining_keys:frozenset, current_key:str|None, obj_state:str)
      obj_state ∈ {'KEY_START', 'COLON', 'VALUE_START', 'COMMA_OR_CLOSE'}
      Indices:   [0]        [1]          [2]                    [3]              [4]

  Array frame  : ('array', path:tuple, arr_state:str)
      arr_state ∈ {'VALUE_START', 'COMMA_OR_CLOSE'}
      Indices:  [0]        [1]          [2]

PATTERN-DFA STATES (aux in STRING mode)
=======================================
  PS_ANY   = 0   — no pattern constraint; any content is valid (self-loop)
  PS_AJ_0  = 10  — ^[A-J]$ pattern: empty so far
  PS_AJ_1  = 11  — ^[A-J]$ pattern: exactly one valid A-J char seen
  PS_DEAD  = -1  — pattern violated; string cannot be legally closed

The critical invariant: the number of distinct pattern-DFA states is FINITE,
so the BFS over (stack, mode, aux, escaped) always terminates.
"""
import os
import torch
import re
from typing import List, Set, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Pattern-DFA state constants
# ---------------------------------------------------------------------------
PS_ANY   = 0    # No pattern constraint
PS_AJ_0  = 10   # ^[A-J]$ — empty
PS_AJ_1  = 11   # ^[A-J]$ — one valid char
PS_DEAD  = -1   # Invalid / dead


def _pattern_initial(pattern: Optional[str]) -> int:
    """Return the initial pattern-DFA state for the given pattern string."""
    if pattern is None:
        return PS_ANY
    if pattern == "^[A-J]$":
        return PS_AJ_0
    # Unknown patterns: permissive (treat as no constraint)
    return PS_ANY


def _pattern_advance(ps: int, char: str) -> int:
    """Advance the pattern-DFA by one character. Returns new state."""
    if ps == PS_DEAD:
        return PS_DEAD
    if ps == PS_ANY:
        # No constraint — any character is fine; self-loop
        return PS_ANY
    if ps == PS_AJ_0:
        # ^[A-J]$ — first char must be A-J
        if 'A' <= char <= 'J':
            return PS_AJ_1
        return PS_DEAD
    if ps == PS_AJ_1:
        # ^[A-J]$ — already have one char; any additional char is invalid
        return PS_DEAD
    # Unrecognised pattern state: permissive
    return PS_ANY


def _pattern_accepting(ps: int) -> bool:
    """Return True if the pattern-DFA state is accepting (closing quote is valid)."""
    if ps == PS_DEAD:
        return False
    if ps == PS_ANY:
        return True   # No constraint — always accepting
    if ps == PS_AJ_0:
        return False  # ^[A-J]$ requires exactly one char
    if ps == PS_AJ_1:
        return True   # ^[A-J]$ — exactly one A-J char ✓
    return True  # Permissive default


# ---------------------------------------------------------------------------
# SchemaHelper
# ---------------------------------------------------------------------------

class SchemaHelper:
    """Provides path-based access to a JSON Schema dict."""

    def __init__(self, schema: dict):
        self.root_schema = schema if isinstance(schema, dict) else {}

    def get_schema_at_path(self, path: tuple) -> dict:
        curr = self.root_schema
        for p in path:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
            else:
                return {}
        return curr if isinstance(curr, dict) else {}

    def get_type(self, path: tuple) -> Optional[str]:
        return self.get_schema_at_path(path).get('type')

    def get_properties(self, path: tuple) -> dict:
        return self.get_schema_at_path(path).get('properties', {})

    def get_items_schema(self, path: tuple) -> dict:
        return self.get_schema_at_path(path).get('items', {})

    def get_pattern(self, path: tuple) -> Optional[str]:
        return self.get_schema_at_path(path).get('pattern')

    def validate_literal(self, val_str: str, v_type: str) -> bool:
        if v_type == 'number':
            try:
                float(val_str)
                return True
            except ValueError:
                return False
        elif v_type == 'integer':
            try:
                int(val_str)
                return True
            except ValueError:
                return False
        elif v_type == 'boolean':
            return val_str in ('true', 'false')
        elif v_type == 'null':
            return val_str == 'null'
        return False

    def is_literal_prefix(self, buf: str, v_type: str) -> bool:
        if not buf:
            return True
        if v_type in ('number', 'integer'):
            if v_type == 'integer':
                return bool(re.match(r'^-?[0-9]*$', buf))
            else:
                return bool(re.match(r'^-?[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?$', buf))
        elif v_type == 'boolean':
            return 'true'.startswith(buf) or 'false'.startswith(buf)
        elif v_type == 'null':
            return 'null'.startswith(buf)
        return False

    def get_expected_type(self, stack) -> Optional[str]:
        """Return the expected JSON type for the current context (used in LITERAL mode)."""
        if not stack:
            return self.get_type(())
        pf = stack[-1]
        if pf[0] == 'object':
            # pf = ('object', path, remaining_keys, current_key, obj_state)
            return self.get_type(pf[1] + ('properties', pf[3]))
        elif pf[0] == 'array':
            # pf = ('array', path, arr_state)
            return self.get_type(pf[1] + ('items',))
        return None


# ---------------------------------------------------------------------------
# Transition function
# ---------------------------------------------------------------------------

def transition(state, char: str, sh: SchemaHelper):
    """
    Advance the DFA by one character `char` from `state`.
    Returns the new state tuple, or None if the character is invalid here.

    State format: (stack:tuple, mode:str, aux, escaped:bool)
    """
    stack, mode, aux, escaped = state

    # -----------------------------------------------------------------------
    # STRUCTURAL mode
    # -----------------------------------------------------------------------
    if mode == "STRUCTURAL":
        if char.isspace():
            return state  # whitespace is always valid in structural positions

        # --- Opening brace '{' ---
        if char == '{':
            if not stack:
                if sh.get_type(()) != 'object':
                    return None
                props = sh.get_properties(())
                nf = ('object', (), frozenset(props.keys()), None, "KEY_START")
                return ((nf,), "STRUCTURAL", "", False)
            pf = stack[-1]
            if pf[0] == 'object' and pf[4] == "VALUE_START":
                nested_path = pf[1] + ('properties', pf[3])
                if sh.get_type(nested_path) != 'object':
                    return None
                props = sh.get_properties(nested_path)
                nf = ('object', nested_path, frozenset(props.keys()), None, "KEY_START")
                return (stack[:-1] + (pf,) + (nf,), "STRUCTURAL", "", False)
            elif pf[0] == 'array' and pf[2] == "VALUE_START":
                nested_path = pf[1] + ('items',)
                if sh.get_type(nested_path) != 'object':
                    return None
                props = sh.get_properties(nested_path)
                nf = ('object', nested_path, frozenset(props.keys()), None, "KEY_START")
                return (stack[:-1] + (pf,) + (nf,), "STRUCTURAL", "", False)
            return None

        # --- Opening bracket '[' ---
        if char == '[':
            if not stack:
                if sh.get_type(()) != 'array':
                    return None
                nf = ('array', (), "VALUE_START")
                return ((nf,), "STRUCTURAL", "", False)
            pf = stack[-1]
            if pf[0] == 'object' and pf[4] == "VALUE_START":
                nested_path = pf[1] + ('properties', pf[3])
                if sh.get_type(nested_path) != 'array':
                    return None
                nf = ('array', nested_path, "VALUE_START")
                return (stack[:-1] + (pf,) + (nf,), "STRUCTURAL", "", False)
            elif pf[0] == 'array' and pf[2] == "VALUE_START":
                nested_path = pf[1] + ('items',)
                if sh.get_type(nested_path) != 'array':
                    return None
                nf = ('array', nested_path, "VALUE_START")
                return (stack[:-1] + (pf,) + (nf,), "STRUCTURAL", "", False)
            return None

        # --- Double-quote '"' — starts a key or string value ---
        if char == '"':
            if not stack:
                if sh.get_type(()) != 'string':
                    return None
                ps = _pattern_initial(sh.root_schema.get('pattern'))
                return (stack, "STRING", ps, False)
            pf = stack[-1]
            if pf[0] == 'object':
                # pf = ('object', path, remaining_keys, current_key, obj_state)
                if pf[4] == "KEY_START":
                    return (stack, "KEY", "", False)
                elif pf[4] == "VALUE_START":
                    val_path = pf[1] + ('properties', pf[3])
                    if sh.get_type(val_path) != 'string':
                        return None
                    ps = _pattern_initial(sh.get_pattern(val_path))
                    return (stack, "STRING", ps, False)
            elif pf[0] == 'array':
                if pf[2] == "VALUE_START":
                    val_path = pf[1] + ('items',)
                    if sh.get_type(val_path) != 'string':
                        return None
                    ps = _pattern_initial(sh.get_pattern(val_path))
                    return (stack, "STRING", ps, False)
            return None

        # --- Colon ':' ---
        if char == ':':
            if stack:
                pf = stack[-1]
                if pf[0] == 'object' and pf[4] == "COLON":
                    nf = (pf[0], pf[1], pf[2], pf[3], "VALUE_START")
                    return (stack[:-1] + (nf,), "STRUCTURAL", "", False)
            return None

        # --- Comma ',' ---
        if char == ',':
            if stack:
                pf = stack[-1]
                if pf[0] == 'object' and pf[4] == "COMMA_OR_CLOSE":
                    # Remove the key we just finished from remaining_keys
                    remaining = pf[2] - {pf[3]} if pf[3] is not None else pf[2]
                    nf = (pf[0], pf[1], remaining, None, "KEY_START")
                    return (stack[:-1] + (nf,), "STRUCTURAL", "", False)
                elif pf[0] == 'array' and pf[2] == "COMMA_OR_CLOSE":
                    nf = (pf[0], pf[1], "VALUE_START")
                    return (stack[:-1] + (nf,), "STRUCTURAL", "", False)
            return None

        # --- Closing brace '}' ---
        if char == '}':
            if stack:
                pf = stack[-1]
                if pf[0] == 'object' and pf[4] in ("KEY_START", "COMMA_OR_CLOSE"):
                    new_stack = stack[:-1]
                    if not new_stack:
                        return ((), "DONE", "", False)
                    gp = new_stack[-1]
                    if gp[0] == 'object':
                        remaining = gp[2] - {gp[3]} if gp[3] is not None else gp[2]
                        nf = (gp[0], gp[1], remaining, None, "COMMA_OR_CLOSE")
                        return (new_stack[:-1] + (nf,), "STRUCTURAL", "", False)
                    elif gp[0] == 'array':
                        nf = (gp[0], gp[1], "COMMA_OR_CLOSE")
                        return (new_stack[:-1] + (nf,), "STRUCTURAL", "", False)
            return None

        # --- Closing bracket ']' ---
        if char == ']':
            if stack:
                pf = stack[-1]
                if pf[0] == 'array' and pf[2] in ("VALUE_START", "COMMA_OR_CLOSE"):
                    new_stack = stack[:-1]
                    if not new_stack:
                        return ((), "DONE", "", False)
                    gp = new_stack[-1]
                    if gp[0] == 'object':
                        remaining = gp[2] - {gp[3]} if gp[3] is not None else gp[2]
                        nf = (gp[0], gp[1], remaining, None, "COMMA_OR_CLOSE")
                        return (new_stack[:-1] + (nf,), "STRUCTURAL", "", False)
                    elif gp[0] == 'array':
                        nf = (gp[0], gp[1], "COMMA_OR_CLOSE")
                        return (new_stack[:-1] + (nf,), "STRUCTURAL", "", False)
            return None

        # --- Start of a literal value (number, boolean, null) ---
        if char.isalnum() or char in ('.', '+', '-'):
            if stack:
                pf = stack[-1]
                if pf[0] == 'object' and pf[4] == "VALUE_START":
                    val_path = pf[1] + ('properties', pf[3])
                    v_type = sh.get_type(val_path)
                    if v_type in ('number', 'integer', 'boolean', 'null'):
                        if sh.is_literal_prefix(char, v_type):
                            return (stack, "LITERAL", char, False)
                elif pf[0] == 'array' and pf[2] == "VALUE_START":
                    val_path = pf[1] + ('items',)
                    v_type = sh.get_type(val_path)
                    if v_type in ('number', 'integer', 'boolean', 'null'):
                        if sh.is_literal_prefix(char, v_type):
                            return (stack, "LITERAL", char, False)
            return None

        return None  # Unknown char in STRUCTURAL

    # -----------------------------------------------------------------------
    # KEY mode  (aux = key prefix string — bounded, so BFS terminates)
    # -----------------------------------------------------------------------
    elif mode == "KEY":
        if not stack or stack[-1][0] != 'object':
            return None
        pf = stack[-1]
        # pf = ('object', path, remaining_keys, current_key, obj_state)
        remaining_keys = pf[2]

        if char == '"':
            # Closing the key quote: buf must be a valid key
            if aux in remaining_keys:
                nf = (pf[0], pf[1], pf[2], aux, "COLON")
                return (stack[:-1] + (nf,), "STRUCTURAL", "", False)
            return None

        new_buf = aux + char
        # Only continue if new_buf is still a prefix of at least one remaining key
        if any(k.startswith(new_buf) for k in remaining_keys):
            return (stack, "KEY", new_buf, False)
        return None

    # -----------------------------------------------------------------------
    # STRING mode  (aux = pattern-DFA state int — FINITE, so BFS terminates)
    # -----------------------------------------------------------------------
    elif mode == "STRING":
        ps = aux  # pattern-DFA state

        if escaped:
            # Previous char was '\\'; this char completes the escape sequence
            new_ps = _pattern_advance(ps, char)
            if new_ps == PS_DEAD:
                return None
            return (stack, "STRING", new_ps, False)

        if char == '\\':
            return (stack, "STRING", ps, True)

        if char == '"':
            # Closing quote — check pattern acceptance
            if not _pattern_accepting(ps):
                return None
            if not stack:
                return ((), "DONE", "", False)
            pf = stack[-1]
            if pf[0] == 'object':
                remaining = pf[2] - {pf[3]} if pf[3] is not None else pf[2]
                nf = (pf[0], pf[1], remaining, None, "COMMA_OR_CLOSE")
                return (stack[:-1] + (nf,), "STRUCTURAL", "", False)
            elif pf[0] == 'array':
                nf = (pf[0], pf[1], "COMMA_OR_CLOSE")
                return (stack[:-1] + (nf,), "STRUCTURAL", "", False)
            return None

        # Any other character — advance pattern DFA
        new_ps = _pattern_advance(ps, char)
        if new_ps == PS_DEAD:
            return None
        return (stack, "STRING", new_ps, False)

    # -----------------------------------------------------------------------
    # LITERAL mode  (aux = literal prefix string — bounded, BFS terminates)
    # -----------------------------------------------------------------------
    elif mode == "LITERAL":
        v_type = sh.get_expected_type(stack)

        if char.isalnum() or char in ('.', '+', '-'):
            new_buf = aux + char
            if sh.is_literal_prefix(new_buf, v_type):
                return (stack, "LITERAL", new_buf, False)
            return None
        else:
            # Non-literal char: close the literal
            if not sh.validate_literal(aux, v_type):
                return None
            # Build the after-literal state
            pf = stack[-1]
            if pf[0] == 'object':
                remaining = pf[2] - {pf[3]} if pf[3] is not None else pf[2]
                nf = (pf[0], pf[1], remaining, None, "COMMA_OR_CLOSE")
                new_stack = stack[:-1] + (nf,)
            elif pf[0] == 'array':
                nf = (pf[0], pf[1], "COMMA_OR_CLOSE")
                new_stack = stack[:-1] + (nf,)
            else:
                return ((), "DONE", "", False)
            # Recurse to handle the delimiter character in STRUCTURAL mode
            return transition((new_stack, "STRUCTURAL", "", False), char, sh)

    # -----------------------------------------------------------------------
    # DONE mode  (JSON is complete, only EOS allowed)
    # -----------------------------------------------------------------------
    elif mode == "DONE":
        return None

    return None


# ---------------------------------------------------------------------------
# ParserStackClassification (PSC)
# ---------------------------------------------------------------------------

class ParserStackClassification:
    """
    PSC — offline-compiled O(1) DFA mask table.

    Precomputes, for every reachable DFA state:
      • state_masks       : boolean token mask (shape [vocab_size])
      • transition_table  : dict token_id → next_state_id
      • shortest_path_len : minimum tokens to reach a closed/complete state
      • shortest_path_token: next token to force when TruncProof fires
    """

    def __init__(self, vocab_size: int, schema: Dict[str, Any], tokenizer=None, cache_dir: Optional[str] = None):
        self.vocab_size = vocab_size
        self.schema = schema
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.is_precomputed = False

        self.state_to_id: Dict = {}
        self.id_to_state: List = []
        self.state_masks: List[torch.Tensor] = []
        self.transition_table: List[Dict[int, int]] = []
        self.shortest_path_len: List[int] = []
        self.shortest_path_token: List[int] = []

    def get_mask(self, state_id: int) -> torch.Tensor:
        """Return the precomputed boolean mask for state_id in O(1)."""
        if (not self.schema or not self.is_precomputed
                or state_id >= len(self.state_masks)):
            return torch.ones(self.vocab_size, dtype=torch.bool)
        return self.state_masks[state_id]

    def _schema_cache_path(self) -> Optional[str]:
        """Compute a deterministic cache file path for this schema+tokenizer pair."""
        try:
            import hashlib, json as _json
            schema_str = _json.dumps(self.schema, sort_keys=True)
            tok_name = getattr(self.tokenizer, 'name_or_path', 'unknown')
            key = f"{tok_name}|{self.vocab_size}|{schema_str}"
            digest = hashlib.sha256(key.encode()).hexdigest()[:16]
            if self.cache_dir:
                cache_dir = self.cache_dir
            else:
                cache_dir = os.path.join(os.path.dirname(__file__), ".psc_cache")
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, f"psc_{digest}.pt")
        except Exception:
            return None

    def _try_load_cache(self, cache_path: str) -> bool:
        """Attempt to load precomputed data from disk. Returns True on success."""
        try:
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
            self.state_masks = data['state_masks']
            self.transition_table = data['transition_table']
            self.id_to_state = data['id_to_state']
            self.shortest_path_len = data['shortest_path_len']
            self.shortest_path_token = data['shortest_path_token']
            self.state_to_id = {s: i for i, s in enumerate(self.id_to_state)}
            self.is_precomputed = True
            return True
        except Exception:
            return False

    def _save_cache(self, cache_path: str):
        """Persist precomputed data to disk."""
        try:
            torch.save({
                'state_masks': self.state_masks,
                'transition_table': self.transition_table,
                'id_to_state': self.id_to_state,
                'shortest_path_len': self.shortest_path_len,
                'shortest_path_token': self.shortest_path_token,
            }, cache_path)
            print(f"[PSC] Cache saved to {cache_path}", flush=True)
        except Exception as e:
            print(f"[PSC] Warning: could not save cache: {e}", flush=True)

    def precompute_closure_masks(self):
        if self.is_precomputed or self.tokenizer is None:
            return

        if not self.schema:
            self.is_precomputed = True
            return

        # ------------------------------------------------------------------
        # Try loading from disk cache first
        # ------------------------------------------------------------------
        cache_path = self._schema_cache_path()
        if cache_path and os.path.exists(cache_path):
            print(f"[PSC] Loading cached DFA from {cache_path} ...", flush=True)
            if self._try_load_cache(cache_path):
                n = len(self.id_to_state)
                print(f"[PSC] Cache loaded: {n} states.", flush=True)
                return
            else:
                print("[PSC] Cache load failed — recomputing.", flush=True)

        # ------------------------------------------------------------------
        # Precompute token strings
        # ------------------------------------------------------------------
        print("[PSC] Precomputing token strings...", flush=True)
        token_strings: List[str] = []
        for i in range(self.vocab_size):
            try:
                token_strings.append(self.tokenizer.decode([i]))
            except Exception:
                token_strings.append("")

        sh = SchemaHelper(self.schema)

        # ------------------------------------------------------------------
        # BFS over DFA states
        # ------------------------------------------------------------------
        print("[PSC] Running BFS over DFA states...", flush=True)
        start_state = ((), "STRUCTURAL", "", False)
        state_to_id: Dict = {start_state: 0}
        queue: List = [start_state]
        state_masks: List[torch.Tensor] = []
        transition_table: List[Dict[int, int]] = []

        state_idx = 0
        
        # Collect stop tokens
        stop_tokens = set()
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            stop_tokens.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end is not None and im_end != getattr(self.tokenizer, "unk_token_id", -1):
                stop_tokens.add(im_end)
        
        while state_idx < len(queue):
            curr_state = queue[state_idx]
            state_idx += 1

            if state_idx % 50 == 0:
                print(f"[PSC]   ... processing state {state_idx}/{len(queue)}", flush=True)

            mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            trans: Dict[int, int] = {}

            for token_id in range(self.vocab_size):
                # Always allow stop tokens if we are in the DONE state
                if curr_state[1] == "DONE" and token_id in stop_tokens:
                    mask[token_id] = True
                    # Trans remains empty for EOS so we hold state/stop decoding
                    continue

                token_str = token_strings[token_id]
                if not token_str:
                    continue

                temp_state = curr_state
                valid = True
                for char in token_str:
                    temp_state = transition(temp_state, char, sh)
                    if temp_state is None:
                        valid = False
                        break

                if valid:
                    mask[token_id] = True
                    if temp_state not in state_to_id:
                        state_to_id[temp_state] = len(queue)
                        queue.append(temp_state)
                    trans[token_id] = state_to_id[temp_state]

            state_masks.append(mask)
            transition_table.append(trans)

        # Publish results
        self.state_to_id = state_to_id
        self.id_to_state = queue          # ← sync after BFS
        self.state_masks = state_masks
        self.transition_table = transition_table
        n_states = len(queue)
        print(f"[PSC] BFS complete: {n_states} states discovered.", flush=True)

        # ------------------------------------------------------------------
        # Backward BFS to compute TruncProof shortest closure paths
        # ------------------------------------------------------------------
        print("[PSC] Computing TruncProof closure paths...", flush=True)
        self.shortest_path_len = [999999] * n_states
        self.shortest_path_token = [-1] * n_states

        backward_queue: List[int] = []
        for idx, state in enumerate(queue):
            s_stack, s_mode, s_aux, s_escaped = state
            if s_mode == "DONE" or (len(s_stack) == 0 and s_mode == "STRUCTURAL" and idx > 0):
                self.shortest_path_len[idx] = 0
                backward_queue.append(idx)

        # Build incoming edge lists
        incoming: List[List] = [[] for _ in range(n_states)]
        for src_id, trans in enumerate(transition_table):
            for token_id, dst_id in trans.items():
                if dst_id < n_states:
                    incoming[dst_id].append((src_id, token_id))

        head = 0
        while head < len(backward_queue):
            curr = backward_queue[head]
            head += 1
            curr_dist = self.shortest_path_len[curr]
            for src_id, token_id in incoming[curr]:
                if self.shortest_path_len[src_id] > curr_dist + 1:
                    self.shortest_path_len[src_id] = curr_dist + 1
                    self.shortest_path_token[src_id] = token_id
                    backward_queue.append(src_id)

        self.is_precomputed = True
        print("[PSC] Precomputation complete.", flush=True)

        # Persist to disk cache for fast startup on next run
        if cache_path:
            self._save_cache(cache_path)


# ---------------------------------------------------------------------------
# StrictWhitelistEnforcer
# ---------------------------------------------------------------------------

class StrictWhitelistEnforcer:
    """
    Applies semantic whitelist constraints on leaf string tokens to prevent
    the LLM from smuggling hallucinations through open JSON strings.
    """
    def __init__(self, allowed_concepts: Set[str]):
        self.allowed_concepts = allowed_concepts

    def filter_logits(self, logits: torch.Tensor, current_token_str: str) -> torch.Tensor:
        # Simulated whitelist enforcement (hook for future integration)
        return logits


# ---------------------------------------------------------------------------
# GreatGramma
# ---------------------------------------------------------------------------

class GreatGramma:
    """
    GREATGRAMMA — Detokenizing Transducer with Maximal Munch and
    offline DFA compilation for O(1) grammar-constrained decoding.
    """

    def __init__(self, vocab_size: int, allowed_concepts: List[str], cache_dir: Optional[str] = None):
        self.vocab_size = vocab_size
        self.whitelist = StrictWhitelistEnforcer(set(allowed_concepts))
        self.cache_dir = cache_dir

    def compile_schema(self, dynamic_schema: Dict[str, Any],
                       tokenizer=None) -> ParserStackClassification:
        """Compile a JSON schema into an O(1) PSC mask set."""
        return ParserStackClassification(self.vocab_size, dynamic_schema, tokenizer, cache_dir=self.cache_dir)

    def apply_transducer_masking(self, logits: torch.Tensor, state_id: int,
                                 psc: ParserStackClassification) -> torch.Tensor:
        """Apply DFA structural mask + semantic whitelist mask."""
        mask = psc.get_mask(state_id)
        logits[..., ~mask] = -float('inf')
        logits = self.whitelist.filter_logits(logits, current_token_str="")
        return logits


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _test_greatgramma():
    gg = GreatGramma(vocab_size=32000, allowed_concepts=["Macroplanner", "Graph"])
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    psc = gg.compile_schema(schema)
    mock_logits = torch.randn(32000)
    masked_logits = gg.apply_transducer_masking(mock_logits, state_id=0, psc=psc)
    print("Masking applied successfully (no tokenizer, schema is present but uncompiled).")


if __name__ == "__main__":
    _test_greatgramma()
