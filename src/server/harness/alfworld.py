"""
ALFWorld Harness — H0/H2/H3/H4/H5/H6

H0  Task Parser        — one-time structured task parsing per episode
H2  Action Gate        — verb-safe canonicalization + invalid blocking
H3  Tool Description   — task-type-aware tool hint embedding
H4  Process Monitor    — world model, subgoal state machine, stall detection
H5  Goal-directed Hint — cold-start skill + per-step guided suggestions
H6  Budget Manager     — step-budget urgency injection + forced completion

Generalization notes:
  - _SEARCH_PRIORITY is based on household commonsense, NOT training statistics.
  - H5 step guidance uses "suggestion" language; hard-forcing only in H6 when
    world-model evidence is unambiguous (item in hand + put action in admissible).
  - All observation/action patterns match the ALFWorld environment API, which is
    identical across train/test splits.
"""

import copy
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_to_word_budget(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).strip()


def _verb_of(action: str) -> str:
    """Return the first word (verb) of an action string."""
    s = action.strip().lower()
    return s.split()[0] if s else ""


def _extract_objects(observation: str) -> List[str]:
    """Extract 'word number' object tokens from an observation string (fixed regex)."""
    if not observation:
        return []
    return re.findall(r"\b([a-z]+\s+\d+)\b", observation.lower())


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retrieval helpers (used by H5 cold-start)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _skill_document_tokens(skill: Dict[str, Any]) -> List[str]:
    parts = [skill.get("text", "")] + list(skill.get("keywords", []))
    return _tokenize(" ".join(parts))


def _bm25_scores(
    query_tokens: List[str],
    docs: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    if not docs:
        return []
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / n_docs
    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    scores: List[float] = []
    unique_q = list(dict.fromkeys(query_tokens))
    for doc in docs:
        dl = len(doc)
        tf: Dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in unique_q:
            if t not in tf:
                continue
            idf = math.log(1.0 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            f = tf[t]
            denom = f + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * f * (k1 + 1.0) / denom
        scores.append(score)
    return scores


def retrieve_skills_for_task(
    task_type: str,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Two-layer skill retrieval:
      1. Hard filter  — keep only skills whose task_types include task_type.
      2. BM25 ranking — rank filtered candidates by relevance to query.
    Returns up to top_k results.
    """
    candidates = [s for s in ALF_SKILLS if task_type in s.get("task_types", [])]
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates   # no need to rank if fits within budget
    query_tokens = _tokenize(query)
    if not query_tokens:
        return candidates[:top_k]
    docs = [_skill_document_tokens(s) for s in candidates]
    scores = _bm25_scores(query_tokens, docs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ALFWorldHarnessConfig:
    enabled: bool = False
    h2_enabled: bool = True
    h3_enabled: bool = True
    h4_enabled: bool = True
    h5_enabled: bool = True
    h6_enabled: bool = True          # new

    # H2
    action_similarity_threshold: float = 0.55   # raised from 0.20
    never_block_prefixes: Tuple[str, ...] = ("go to ", "open ", "close ")
    invalid_block_after: int = 2

    # H3
    h3_max_words: int = 20

    # H4
    h4_stall_window: int = 4
    h4_soft_intervention_rounds: int = 2
    h4_min_rounds_before_stall: int = 8
    h4_post_put_grace: int = 4       # new: rounds after PUT where stall is suppressed

    # H5
    h5_top_k: int = 1          # max skills injected at cold-start
    h5_cold_start_max_words: int = 35     # word cap per cold-start skill
    h5_step_hint_max_words: int = 25      # word cap for per-step guidance

    # H4 — extra stall patterns
    h4_nothing_happens_window: int = 4      # obs window for "nothing happens" loop check
    h4_nothing_happens_threshold: int = 3   # how many "nothing happens" obs to trigger

    # H2 — empty-turn intervention
    h2_empty_turn_threshold: int = 2        # consecutive empty actions before reminder

    # H6
    h6_warn_threshold: int = 7       # remaining steps < N → inject urgency hint
    h6_force_threshold: int = 4      # remaining steps < N → force PUT if possible


# ─────────────────────────────────────────────────────────────────────────────
# H0 — Task Context
# ─────────────────────────────────────────────────────────────────────────────

# Subgoal labels
_SG_FIND      = "FIND"
_SG_TAKE      = "TAKE"
_SG_GOTO_SINK = "GOTO_SINK"
_SG_CLEAN     = "CLEAN"
_SG_GOTO_MCW  = "GOTO_MICROWAVE"
_SG_HEAT      = "HEAT"
_SG_GOTO_FRG  = "GOTO_FRIDGE"
_SG_COOL      = "COOL"
_SG_GOTO_DEST = "GOTO_DEST"
_SG_PUT       = "PUT"
_SG_GOTO_LAMP = "GOTO_LAMP"
_SG_USE_LAMP  = "USE_LAMP"
_SG_EXAMINE   = "EXAMINE"
_SG_DONE      = "DONE"

_SUBGOAL_CHAINS: Dict[str, List[str]] = {
    "pick_and_place":        [_SG_FIND, _SG_TAKE, _SG_GOTO_DEST, _SG_PUT],
    "pick_clean_then_place": [_SG_FIND, _SG_TAKE, _SG_GOTO_SINK, _SG_CLEAN, _SG_GOTO_DEST, _SG_PUT],
    "pick_heat_then_place":  [_SG_FIND, _SG_TAKE, _SG_GOTO_MCW,  _SG_HEAT,  _SG_GOTO_DEST, _SG_PUT],
    "pick_cool_then_place":  [_SG_FIND, _SG_TAKE, _SG_GOTO_FRG,  _SG_COOL,  _SG_GOTO_DEST, _SG_PUT],
    # look_at_obj: lamp must be ON before examine action appears; agent must also
    # TAKE the target (examine X with desklamp only shows in admissible when holding X).
    "look_at_obj":           [_SG_GOTO_LAMP, _SG_USE_LAMP, _SG_FIND, _SG_TAKE, _SG_EXAMINE],
    # pick_two_obj: two full pick-place cycles (agent can only hold one item at a time)
    "pick_two_obj":          [
        _SG_FIND, _SG_TAKE, _SG_GOTO_DEST, _SG_PUT,
        _SG_FIND, _SG_TAKE, _SG_GOTO_DEST, _SG_PUT,
    ],
}

# Transformation keyword → intermediate location name
_XFORM_LOCATION: Dict[str, str] = {
    "clean": "sinkbasin",
    "heat":  "microwave",
    "cool":  "fridge",
}

# Search priority: household commonsense, not derived from training statistics.
# Only used as a soft hint for exploration ordering, never as a hard force.
_SEARCH_PRIORITY: Dict[str, List[str]] = {
    "kettle":      ["stoveburner", "cabinet", "countertop", "microwave"],
    "pan":         ["stoveburner", "cabinet", "countertop"],
    "pot":         ["stoveburner", "cabinet", "countertop"],
    "plate":       ["cabinet", "countertop", "shelf", "drawer"],
    "bowl":        ["cabinet", "countertop", "shelf", "drawer"],
    "cup":         ["cabinet", "countertop", "shelf", "drawer"],
    "mug":         ["cabinet", "shelf", "countertop", "desk"],
    "fork":        ["drawer", "countertop", "diningtable"],
    "knife":       ["drawer", "countertop", "diningtable"],
    "spoon":       ["drawer", "countertop", "diningtable"],
    "butterknife": ["drawer", "countertop", "diningtable"],
    "spatula":     ["drawer", "countertop", "diningtable"],
    "newspaper":   ["coffeetable", "diningtable", "sidetable", "sofa", "ottoman"],
    "book":        ["shelf", "sidetable", "coffeetable"],
    "handtowel":   ["handtowelholder", "cabinet", "countertop"],
    "cloth":       ["handtowelholder", "countertop", "cabinet"],
    "soapbar":     ["bathtubbasin", "sinkbasin", "cabinet", "shelf"],
    "toiletpaper": ["toiletpaperhanger", "cabinet", "shelf"],
    "keychain":    ["sidetable", "drawer", "coffeetable"],
    "cellphone":   ["sidetable", "drawer", "coffeetable", "desk"],
    "creditcard":  ["sidetable", "drawer", "coffeetable"],
    "pencil":      ["desk", "drawer", "sidetable"],
    "pen":         ["desk", "drawer", "sidetable"],
    "cd":          ["desk", "shelf", "drawer"],
    "tomato":      ["fridge", "countertop", "diningtable"],
    "apple":       ["fridge", "countertop", "diningtable"],
    "bread":       ["cabinet", "countertop", "microwave"],
    "egg":         ["fridge", "countertop"],
    "lettuce":     ["fridge", "countertop", "diningtable"],
    "potato":      ["fridge", "countertop", "diningtable"],
    "dishsponge":  ["sinkbasin", "cabinet", "countertop"],
    "glassbottle": ["cabinet", "countertop", "fridge"],
    "remotecontrol": ["sofa", "sidetable", "coffeetable", "ottoman"],
}
_SEARCH_PRIORITY_DEFAULT = [
    "countertop", "cabinet", "shelf", "drawer",
    "diningtable", "sidetable", "coffeetable",
    "desk", "fridge", "sofa",
]


@dataclass
class TaskContext:
    task_type: str
    target_type: str               # e.g. "kettle"
    destination_type: str          # e.g. "countertop"
    transformation: Optional[str]  # "clean" / "heat" / "cool" / None
    count: int                     # 1 or 2
    subgoal_chain: List[str]
    task_goal: str                 # raw task description, used as BM25 query


def _detect_task_type(desc: str) -> str:
    d = desc.lower()
    if re.search(r"\bclean\b", d):
        return "pick_clean_then_place"
    if re.search(r"\bheat\b|\bhot\b", d):
        return "pick_heat_then_place"
    if re.search(r"\bcool\b|\bcold\b", d):
        return "pick_cool_then_place"
    # ALFWorld uses two phrasings for look_at_obj:
    #   "examine the X with the desklamp"
    #   "look at X under the desklamp"
    if re.search(r"\bexamine\b|\blook\s+at\b", d):
        return "look_at_obj"
    if re.search(r"\btwo\b", d):
        return "pick_two_obj"
    return "pick_and_place"


def _extract_target_dest(task_desc: str) -> Tuple[str, str]:
    d = task_desc.lower()
    for pfx in ("your task is to:", "here is your task.", "here is your task:"):
        if pfx in d:
            d = d.split(pfx, 1)[-1].strip()
    d = d.rstrip(".").strip()

    # "examine the X [Y] with the desklamp"  — one phrasing used by ALFWorld
    m = re.search(r"examine\s+the\s+([\w]+(?:\s+[\w]+)?)\s+with", d)
    if m:
        return m.group(1).strip(), "desklamp"

    # "look at X under the desklamp"  — alternate phrasing used by ALFWorld
    m = re.search(r"look\s+at\s+([\w]+(?:\s+[\w]+)?)\s+under", d)
    if m:
        return m.group(1).strip(), "desklamp"

    # "find two/a/some X and put them/it in/on Y"
    m = re.search(
        r"find\s+(?:two|a|some)\s+(\w+)\s+and\s+put\s+(?:it|them)\s+(?:in|on)\s+(\w+)", d
    )
    if m:
        # strip plural 's' for pick_two_obj (e.g. "pillows" → "pillow")
        target = m.group(1)
        if target.endswith("s") and len(target) > 3:
            target = target[:-1]
        return target, m.group(2)

    # "clean/heat/cool some X and put it in/on Y"
    m = re.search(
        r"(?:clean|heat|cool)\s+some\s+(\w+)\s+and\s+put\s+it\s+(?:in|on)\s+(\w+)", d
    )
    if m:
        return m.group(1), m.group(2)

    # "put a/some/two [adjective] X in/on Y"  (adj = hot/cool/cold/clean/heated/cooled)
    m = re.search(
        r"put\s+(?:a|some|two)\s+(?:hot|cool|cold|clean|heated|cooled|cleaned)?\s*(\w+)\s+(?:in|on)\s+(\w+)",
        d,
    )
    if m:
        tgt, dst = m.group(1).strip(), m.group(2).strip()
        if tgt not in ("it", "them", "and"):
            return tgt, dst

    return "", ""


def parse_task_context(init_prompt: str) -> TaskContext:
    """H0: Parse the episode's init_prompt into a structured TaskContext."""
    m = re.search(r"[Yy]our task is to:\s*(.+?)(?:\.?\s*AVAILABLE|\.|$)", init_prompt)
    task_goal = m.group(1).strip() if m else init_prompt

    task_type = _detect_task_type(task_goal)
    target_type, destination_type = _extract_target_dest(task_goal)
    transformation = {
        "pick_clean_then_place": "clean",
        "pick_heat_then_place":  "heat",
        "pick_cool_then_place":  "cool",
    }.get(task_type)
    count = 2 if task_type == "pick_two_obj" else 1
    chain = list(_SUBGOAL_CHAINS.get(task_type, [_SG_FIND, _SG_TAKE, _SG_GOTO_DEST, _SG_PUT]))
    return TaskContext(
        task_type=task_type,
        target_type=target_type,
        destination_type=destination_type,
        transformation=transformation,
        count=count,
        subgoal_chain=chain,
        task_goal=task_goal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# H4 — World Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorldModel:
    """Incremental spatial knowledge built from observations and actions."""
    object_at: Dict[str, str] = field(default_factory=dict)
    inventory: Optional[str] = None
    visited: set = field(default_factory=set)
    unvisited: List[str] = field(default_factory=list)
    current_location: Optional[str] = None
    target_found: bool = False
    target_location: Optional[str] = None
    placed_count: int = 0
    placed_locations: List[str] = field(default_factory=list)  # destination locs used so far

    def update(
        self,
        action: str,
        observation: str,
        admissible: List[str],
        target_type: str,
    ) -> None:
        obs_lower = (observation or "").lower()
        act_lower = (action or "").strip().lower()

        # Track current location from navigation actions
        goto_m = re.match(r"go to (.+)", act_lower)
        if goto_m:
            loc = goto_m.group(1).strip()
            self.current_location = loc
            self.visited.add(loc)
            if loc in self.unvisited:
                self.unvisited.remove(loc)

        # Inventory: "you pick up the X from"
        pick_m = re.search(r"you pick up the (.+?) from", obs_lower)
        if pick_m:
            self.inventory = pick_m.group(1).strip()

        # Inventory: "you put the X in/on the Y" → drop item
        if act_lower.startswith("put ") and re.search(r"you put the .+? (?:in|on) ", obs_lower):
            self.inventory = None
            self.placed_count += 1
            # Record where we just placed an item so future hints can avoid pointing
            # back to this location as a search target (relevant for pick_two_obj).
            if self.current_location and self.current_location not in self.placed_locations:
                self.placed_locations.append(self.current_location)

        # Parse objects visible at current location
        if self.current_location:
            for obj in _extract_objects(observation):
                if obj not in self.object_at:
                    self.object_at[obj] = self.current_location
                if target_type and not self.target_found and target_type.lower() in obj:
                    self.target_found = True
                    self.target_location = self.current_location

        # Extend unvisited from "go to X" entries in admissible
        for cmd in admissible:
            if cmd.startswith("go to "):
                loc = cmd[len("go to "):]
                if loc not in self.visited and loc not in self.unvisited:
                    self.unvisited.append(loc)

    def ordered_unvisited(self, target_type: str) -> List[str]:
        """Unvisited locations sorted by commonsense priority for target_type."""
        priority = _SEARCH_PRIORITY.get(target_type.lower(), _SEARCH_PRIORITY_DEFAULT)

        def rank(loc: str) -> int:
            ll = loc.lower()
            for i, p in enumerate(priority):
                if ll.startswith(p):
                    return i
            return len(priority)

        return sorted(self.unvisited, key=rank)

    def find_take_action(self, target_type: str, admissible: List[str]) -> Optional[str]:
        for a in admissible:
            if a.startswith("take ") and target_type.lower() in a.lower():
                return a
        return None

    def find_put_action(self, dest_type: str, admissible: List[str]) -> Optional[str]:
        """Return the put action for dest_type, or None.

        No fallback: returning a wrong-destination put action causes the agent
        to place the item in the wrong receptacle, which then can't be undone
        and triggers a "Nothing happens" loop at the correct destination.
        """
        if not self.inventory:
            return None
        for a in admissible:
            if a.startswith("put ") and dest_type.lower() in a.lower():
                return a
        return None

    def find_transform_action(self, transform: str, admissible: List[str]) -> Optional[str]:
        for a in admissible:
            if a.startswith(transform + " "):
                return a
        return None

    def find_examine_action(self, target_type: str, admissible: List[str]) -> Optional[str]:
        for a in admissible:
            if "examine" in a and target_type.lower() in a.lower():
                return a
        return None

    def find_use_lamp_action(self, admissible: List[str]) -> Optional[str]:
        for a in admissible:
            if a.startswith("use desklamp") or ("desklamp" in a and "use" in a):
                return a
        return None


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Action Gate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gate_action(
    action: str,
    admissible: List[str],
    threshold: float,
) -> Tuple[Optional[str], bool, float]:
    """
    Verb-safe similarity matching.

    Returns (best_candidate, canonicalized, score).
    Only considers candidates that share the same verb as `action`.
    If no same-verb candidates exist, returns (admissible[0], False, 0.0) —
    i.e., never substitutes across different verbs (e.g. take ↔ put).
    """
    verb = _verb_of(action)
    same_verb = [a for a in admissible if _verb_of(a) == verb]

    # Hard guard: if no candidate shares the verb, refuse to canonicalize.
    if not same_verb:
        return admissible[0], False, 0.0

    best_action = same_verb[0]
    best_score = -1.0
    for candidate in same_verb:
        score = SequenceMatcher(None, action, candidate).ratio()
        if score > best_score:
            best_score = score
            best_action = candidate

    canonicalized = best_score >= threshold
    return best_action, canonicalized, best_score


# ─────────────────────────────────────────────────────────────────────────────
# H4 — Subgoal advancement helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_forced_action(admissible: List[str], last_output: str) -> Optional[str]:
    last_output = (last_output or "").strip().lower()
    for action in admissible:
        low = action.lower()
        if low == last_output:
            continue
        if low.startswith("inventory") or low.startswith("look"):
            continue
        return action
    return admissible[0] if admissible else None


# ─────────────────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ALFWorldHarnessRuntime:
    config: ALFWorldHarnessConfig

    # H2 state
    invalid_consecutive_count: int = 0
    force_next_action: Optional[str] = None
    empty_turn_count: int = 0          # P2-1: consecutive empty-action turns

    # H4 state
    last_outputs: List[str] = field(default_factory=list)
    last_observations: List[str] = field(default_factory=list)  # P1-1: for nothing-happens check
    soft_intervention_count: int = 0
    post_put_grace: int = 0

    # H0 / H4 world state
    task_ctx: Optional[TaskContext] = field(default=None)
    world: WorldModel = field(default_factory=WorldModel)
    _subgoal_idx: int = field(default=0)

    # H5 step-guidance dedup state
    _last_step_hint: Optional[str] = field(default=None)
    _last_step_hint_sg_idx: int = field(default=-1)

    # ── H0 ───────────────────────────────────────────────────────────────────

    def init_task(self, init_prompt: str, initial_admissible: List[str]) -> None:
        """Parse task context and seed the world model with initial admissible locations."""
        self.task_ctx = parse_task_context(init_prompt)
        self._subgoal_idx = 0
        self.world = WorldModel()
        # Seed unvisited from initial navigable locations
        for cmd in initial_admissible:
            if cmd.startswith("go to "):
                loc = cmd[len("go to "):]
                if loc not in self.world.unvisited:
                    self.world.unvisited.append(loc)

        # P2-2: If the target is immediately takeable from the start (visible in initial
        # admissible), mark it found so the first FIND hint is actionable right away.
        if self.task_ctx.target_type:
            for cmd in initial_admissible:
                if (cmd.startswith("take ")
                        and self.task_ctx.target_type.lower() in cmd.lower()):
                    self.world.target_found = True
                    self.world.target_location = "starting location"
                    break

    def _current_subgoal(self) -> str:
        if self.task_ctx is None:
            return _SG_DONE
        chain = self.task_ctx.subgoal_chain
        if self._subgoal_idx < len(chain):
            return chain[self._subgoal_idx]
        return _SG_DONE

    def _advance_subgoal(self, action: str, observation: str, admissible: List[str]) -> None:
        """Advance the subgoal state machine based on post-step evidence."""
        if self.task_ctx is None:
            return
        sg = self._current_subgoal()
        obs_lower = (observation or "").lower()
        tt = self.task_ctx.target_type
        dt = self.task_ctx.destination_type

        if sg == _SG_FIND:
            # Advance when "take target" is available — applies to ALL task types.
            # For look_at_obj, the agent must take the object before examine X with desklamp
            # appears in admissible (lamp can be on but examine only shows up when holding X).
            if self.world.find_take_action(tt, admissible):
                self._subgoal_idx += 1

        elif sg == _SG_TAKE:
            if "you pick up" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_GOTO_SINK:
            if self.world.find_transform_action("clean", admissible):
                self._subgoal_idx += 1

        elif sg == _SG_CLEAN:
            if "you clean" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_GOTO_MCW:
            if self.world.find_transform_action("heat", admissible):
                self._subgoal_idx += 1

        elif sg == _SG_HEAT:
            if "you heat" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_GOTO_FRG:
            if self.world.find_transform_action("cool", admissible):
                self._subgoal_idx += 1

        elif sg == _SG_COOL:
            if "you cool" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_GOTO_DEST:
            if "you put" in obs_lower:
                # PUT was already executed while we were in GOTO_DEST state.
                # Advance past both GOTO_DEST and PUT in one step.
                self._subgoal_idx += 2
                self.post_put_grace = self.config.h4_post_put_grace
                # P0-2 (same as _SG_PUT case): if next subgoal is FIND, reset target state.
                if self._current_subgoal() == _SG_FIND:
                    self.world.target_found = False
                    self.world.target_location = None
            elif self.world.find_put_action(dt, admissible):
                self._subgoal_idx += 1  # → PUT (destination reached, ready to put)

        elif sg == _SG_PUT:
            if "you put" in obs_lower:
                self._subgoal_idx += 1
                self.post_put_grace = self.config.h4_post_put_grace
                # P0-2: If the next subgoal is FIND (pick_two_obj second cycle),
                # reset target search state so we don't navigate back to the now-empty
                # source location. Generalises to any future task with multi-cycle PUT→FIND.
                if self._current_subgoal() == _SG_FIND:
                    self.world.target_found = False
                    self.world.target_location = None

        elif sg == _SG_GOTO_LAMP:
            if self.world.find_use_lamp_action(admissible):
                self._subgoal_idx += 1

        elif sg == _SG_USE_LAMP:
            if "you turn on" in obs_lower or "you switch on" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_EXAMINE:
            # Terminal — task completion detected by done flag in alfworld_run
            pass

    # ── H3 ───────────────────────────────────────────────────────────────────

    def build_h3_hint(self) -> str:
        if self.task_ctx is None:
            return _truncate_to_word_budget(
                "Always call take_action with exactly one action from AVAILABLE ACTIONS.",
                self.config.h3_max_words,
            )
        hints = {
            "pick_and_place":        "Prioritize: locate target → take → navigate to destination → put.",
            "pick_two_obj":          "You can carry one item at a time. Deliver first, then collect second.",
            "pick_clean_then_place": "Locate target → take → go to sinkbasin → clean → go to destination → put.",
            "pick_heat_then_place":  "Locate target → take → go to microwave → heat → go to destination → put.",
            "pick_cool_then_place":  "Locate target → take → go to fridge → cool → go to destination → put.",
            "look_at_obj":           "Go to desklamp → use it → find target object → take it → examine with desklamp.",
        }
        hint = hints.get(self.task_ctx.task_type, hints["pick_and_place"])
        return _truncate_to_word_budget(hint, self.config.h3_max_words)

    # ── H2 ───────────────────────────────────────────────────────────────────

    def pre_validate_action(self, raw_action: str, admissible: List[str]) -> Dict[str, Any]:
        # Apply forced action from H4 hard intervention — but yield to task-critical agent actions
        if self.force_next_action:
            agent_verb = _verb_of((raw_action or "").strip().lower())
            critical = {"take", "put", "clean", "heat", "cool", "use", "examine"}
            agent_in_admissible = (raw_action or "").strip().lower() in admissible
            if agent_verb in critical and agent_in_admissible:
                # Agent is about to do something important; honour it and discard the force
                self.force_next_action = None
            else:
                forced = self.force_next_action
                self.force_next_action = None
                return {
                    "action": forced,
                    "canonicalized": True,
                    "blocked": False,
                    "reason": "h4_forced_action",
                    "raw_action": raw_action,
                }

        action = (raw_action or "").strip().lower().split("\n")[0]
        if not admissible:
            return {
                "action": action,
                "canonicalized": False,
                "blocked": False,
                "reason": "no_admissible_actions",
                "raw_action": raw_action,
            }

        # Exact match
        if action in admissible:
            self.invalid_consecutive_count = 0
            return {
                "action": action,
                "canonicalized": False,
                "blocked": False,
                "reason": "exact_match",
                "raw_action": raw_action,
            }

        # Verb-safe similarity match
        best_action, canonicalized, score = _gate_action(
            action, admissible, self.config.action_similarity_threshold
        )
        chosen = best_action if canonicalized else action
        is_invalid = chosen not in admissible

        if is_invalid:
            self.invalid_consecutive_count += 1
        else:
            self.invalid_consecutive_count = 0

        blockable = not any(action.startswith(p) for p in self.config.never_block_prefixes)
        blocked = (
            is_invalid
            and blockable
            and self.invalid_consecutive_count >= self.config.invalid_block_after
        )
        reason = "canonicalized" if canonicalized else "invalid_action"
        if blocked:
            reason = "blocked_invalid_action"
        return {
            "action": chosen,
            "canonicalized": canonicalized,
            "blocked": blocked,
            "reason": reason,
            "raw_action": raw_action,
        }

    # ── H4 ───────────────────────────────────────────────────────────────────

    def post_step_monitor(
        self,
        raw_output: str,
        final_action: str,
        observation: str,
        admissible: List[str],
    ) -> Dict[str, Any]:
        # Update world model
        if self.task_ctx:
            self.world.update(final_action, observation, admissible, self.task_ctx.target_type)
            self._advance_subgoal(final_action, observation, admissible)

        # Decrement post-PUT grace period
        if self.post_put_grace > 0:
            self.post_put_grace -= 1

        # Track recent outputs/observations for stall detection
        self.last_outputs.append((raw_output or "").strip().lower())
        if len(self.last_outputs) > self.config.h4_stall_window:
            self.last_outputs = self.last_outputs[-self.config.h4_stall_window:]

        obs_lower_str = (observation or "").lower()
        self.last_observations.append(obs_lower_str)
        win = self.config.h4_nothing_happens_window
        if len(self.last_observations) > win:
            self.last_observations = self.last_observations[-win:]

        response: Dict[str, Any] = {
            "stall_score": 0,
            "intervention_level": "none",
            "audit_reason": "",
            "recovery_prompt": None,
            "forced_action": None,
        }

        # P2-1: Empty turn detection — agent produced no tool call / empty action.
        # Inject a nudge after consecutive empty turns to avoid wasting max_step budget.
        if not final_action or not final_action.strip():
            self.empty_turn_count += 1
            if self.empty_turn_count >= self.config.h2_empty_turn_threshold:
                sg_label = self._current_subgoal()
                tt = self.task_ctx.target_type if self.task_ctx else "target"
                response["intervention_level"] = "soft"
                response["audit_reason"] = "empty_turn"
                response["recovery_prompt"] = (
                    f"Harness: you must call take_action with an action from AVAILABLE ACTIONS. "
                    f"Current goal: {sg_label} ({tt}). "
                    f"Choose any listed action to move forward."
                )
                return response
        else:
            self.empty_turn_count = 0

        if not self.config.h4_enabled:
            return response

        # Stall detection: suppressed during grace period or when subgoal is DONE
        in_grace = self.post_put_grace > 0
        sg_done = self._current_subgoal() == _SG_DONE
        if in_grace or sg_done:
            return response

        # P1-1: "Nothing happens" loop — agent is not holding anything but keeps issuing
        # put/take actions that have no effect.  This pattern is missed by the standard
        # repeated_action check because the agent may alternate between navigation and put.
        nothing_happens_count = sum(
            1 for obs in self.last_observations
            if "nothing happens" in obs
        )
        if (nothing_happens_count >= self.config.h4_nothing_happens_threshold
                and self.world.inventory is None):
            tt = self.task_ctx.target_type if self.task_ctx else "the target object"
            response["stall_score"] = 2
            response["intervention_level"] = "soft"
            response["audit_reason"] = "nothing_happens_loop"
            response["recovery_prompt"] = (
                f"Harness: you are not holding anything — put actions have no effect. "
                f"Find and pick up the {tt} before trying to place it."
            )
            # Reset observation window so we don't fire repeatedly on the same episode
            self.last_observations.clear()
            return response

        enough_rounds = len(self.last_outputs) >= self.config.h4_min_rounds_before_stall
        repeated_action = (
            len(self.last_outputs) >= self.config.h4_stall_window
            and len(set(self.last_outputs[-self.config.h4_stall_window:])) == 1
        )

        # Object novelty (now works correctly after regex fix)
        new_objects = set(_extract_objects(observation))
        no_new_objects = len(new_objects) == 0 or new_objects.issubset(
            set(self.world.object_at.keys())
        )

        stall_score = int(repeated_action) + int(no_new_objects)
        response["stall_score"] = stall_score
        stalled = enough_rounds and repeated_action and no_new_objects

        if not stalled:
            return response

        if self.soft_intervention_count < self.config.h4_soft_intervention_rounds:
            self.soft_intervention_count += 1
            response["intervention_level"] = "soft"
            response["audit_reason"] = "stall_detected_soft"
            response["recovery_prompt"] = (
                "Harness: you seem stuck. Choose a different, non-repeated action "
                "from AVAILABLE ACTIONS and avoid repeating look/inventory."
            )
        else:
            forced = _pick_forced_action(admissible, self.last_outputs[-1] if self.last_outputs else "")
            if forced:
                self.force_next_action = forced
                response["intervention_level"] = "hard"
                response["audit_reason"] = "stall_persists"
                response["forced_action"] = forced
                response["recovery_prompt"] = (
                    "Harness: breaking the loop — execute a different exploration action next turn."
                )
        return response

    # ── H5 ───────────────────────────────────────────────────────────────────

    def cold_start_skill_hints(self) -> List[Dict[str, str]]:
        """
        H5 cold-start: two-layer retrieval.
          1. Filter ALF_SKILLS by task_type tag.
          2. BM25-rank filtered candidates against the task goal string.
          3. Return top-k (config.h5_top_k) skills, each capped at max_words.
        """
        if not self.config.h5_enabled or self.task_ctx is None:
            return []
        skills = retrieve_skills_for_task(
            task_type=self.task_ctx.task_type,
            query=self.task_ctx.task_goal,
            top_k=self.config.h5_top_k,
        )
        result = []
        for skill in skills:
            text = _truncate_to_word_budget(skill["text"], self.config.h5_cold_start_max_words)
            result.append({
                "id": skill["id"],
                "text": text,
                "trigger": "cold_start",
                "token_cost": str(len(text.split())),
            })
        return result

    def step_guidance(self, current_round: int, max_step: int, admissible: List[str]) -> Optional[str]:
        """
        H5 per-step guidance: generate a specific, actionable hint based on
        current subgoal state and world model.

        Returns None when no helpful guidance can be produced (to avoid noise).
        Guidance is always phrased as a suggestion, not a command, to preserve
        agent autonomy and generalisation across scenes.
        """
        if not self.config.h5_enabled or self.task_ctx is None:
            return None

        sg = self._current_subgoal()
        tt = self.task_ctx.target_type
        dt = self.task_ctx.destination_type
        xform = self.task_ctx.transformation
        world = self.world

        hint: Optional[str] = None

        if sg == _SG_FIND:
            if self.task_ctx.task_type == "look_at_obj":
                # P0-3: agent must TAKE the target object first (examine X with desklamp
                # only appears in admissible when holding X and the lamp is on).
                if world.target_found and world.target_location:
                    hint = (
                        f"Hint: {tt} was spotted at {world.target_location}. "
                        f"Navigate there and take it."
                    )
                elif world.unvisited:
                    ordered = world.ordered_unvisited(tt)
                    next_loc = ordered[0] if ordered else world.unvisited[0]
                    hint = f"Hint: searching for {tt}. Try: go to {next_loc}."
            else:
                if world.target_found and world.target_location:
                    hint = (
                        f"Hint: {tt} was spotted at {world.target_location}. "
                        f"Navigate there and take it."
                    )
                elif world.unvisited:
                    ordered = world.ordered_unvisited(tt)
                    # P1-2: exclude already-placed destination locations from search hints
                    # (avoids directing agent back to spot where it just deposited an item).
                    filtered = [loc for loc in ordered if loc not in world.placed_locations]
                    next_loc = (filtered[0] if filtered else ordered[0]) if ordered else world.unvisited[0]
                    hint = f"Hint: {tt} not found yet. Suggested next: go to {next_loc}."

        elif sg == _SG_TAKE:
            take_a = world.find_take_action(tt, admissible)
            if take_a:
                hint = f"Hint: pick up the {tt} — use: {take_a}."

        elif sg == _SG_GOTO_SINK:
            if not any(a.startswith("clean ") for a in admissible):
                hint = f"Hint: go to sinkbasin to clean the {world.inventory or tt}."

        elif sg == _SG_CLEAN:
            clean_a = world.find_transform_action("clean", admissible)
            if clean_a:
                hint = f"Hint: clean the {world.inventory or tt} — use: {clean_a}."

        elif sg == _SG_GOTO_MCW:
            if not any(a.startswith("heat ") for a in admissible):
                hint = f"Hint: go to microwave to heat the {world.inventory or tt}."

        elif sg == _SG_HEAT:
            heat_a = world.find_transform_action("heat", admissible)
            if heat_a:
                hint = f"Hint: heat the {world.inventory or tt} — use: {heat_a}."

        elif sg == _SG_GOTO_FRG:
            if not any(a.startswith("cool ") for a in admissible):
                hint = f"Hint: go to fridge to cool the {world.inventory or tt}."

        elif sg == _SG_COOL:
            cool_a = world.find_transform_action("cool", admissible)
            if cool_a:
                hint = f"Hint: cool the {world.inventory or tt} — use: {cool_a}."

        elif sg in (_SG_GOTO_DEST, _SG_PUT):
            put_a = world.find_put_action(dt, admissible)
            if put_a:
                # Already at destination (or PUT is immediately available)
                hint = f"Hint: place the {world.inventory or tt} — use: {put_a}."
            else:
                hint = f"Hint: go to {dt} to place the {world.inventory or tt}."

        elif sg == _SG_GOTO_LAMP:
            hint = "Hint: find the desklamp in this room and go to it."

        elif sg == _SG_USE_LAMP:
            lamp_a = world.find_use_lamp_action(admissible)
            if lamp_a:
                hint = f"Hint: turn on the lamp — use: {lamp_a}."

        elif sg == _SG_EXAMINE:
            ex_a = world.find_examine_action(tt, admissible)
            if ex_a:
                # P0-3: be explicit — agent must use the full "examine X with desklamp" form.
                hint = f"Hint: use exactly: {ex_a}."
            elif world.inventory:
                # Holding the item but examine not yet available — guide back to lamp.
                hint = (
                    f"Hint: you are holding {world.inventory}. "
                    f"Use 'examine {tt} with desklamp' to complete the task."
                )

        if hint is None:
            return None
        hint = _truncate_to_word_budget(hint, self.config.h5_step_hint_max_words)

        # Dedup: suppress if the hint content and subgoal haven't changed since last injection.
        sg_changed = (self._subgoal_idx != self._last_step_hint_sg_idx)
        content_changed = (hint != self._last_step_hint)
        if not sg_changed and not content_changed:
            return None

        self._last_step_hint = hint
        self._last_step_hint_sg_idx = self._subgoal_idx
        return hint

    # ── H6 ───────────────────────────────────────────────────────────────────

    def budget_check(self, remaining_steps: int, admissible: List[str]) -> Dict[str, Any]:
        """
        H6: Step-budget management.

        - Warn when budget is low and we're still searching.
        - Force a PUT action when budget is critical and conditions are unambiguous:
          agent holds the correct item AND the put action is in admissible.
          This is the only place where harness forces a non-exploration action.
        """
        response: Dict[str, Any] = {"hint": None, "force_action": None}
        if not self.config.h6_enabled or self.task_ctx is None:
            return response

        sg = self._current_subgoal()
        world = self.world
        dt = self.task_ctx.destination_type
        tt = self.task_ctx.target_type

        # Hard force: agent holds correct item, put action is available, steps critical
        if remaining_steps < self.config.h6_force_threshold:
            if world.inventory and sg in (_SG_GOTO_DEST, _SG_PUT):
                put_a = world.find_put_action(dt, admissible)
                if put_a:
                    response["force_action"] = put_a
                    response["hint"] = (
                        f"[{remaining_steps} steps left] Harness: placing {world.inventory} now."
                    )
                    return response

        # Soft warn: still searching with few steps left
        if remaining_steps < self.config.h6_warn_threshold and sg == _SG_FIND:
            ordered = world.ordered_unvisited(tt)
            top = ordered[:2] if ordered else []
            if top:
                locs = " or ".join(top)
                response["hint"] = (
                    f"[{remaining_steps} steps left] Urgently check: {locs}."
                )

        return response


# ─────────────────────────────────────────────────────────────────────────────
# H3 helper — tool description patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_take_action_tool_description(
    tools: Optional[List[Dict[str, Any]]], hint: str
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return tools
    patched = copy.deepcopy(tools)
    for tool in patched:
        fn = tool.get("function", {})
        if fn.get("name") != "take_action":
            continue
        description = fn.get("description", "").strip()
        if hint in description:
            break
        fn["description"] = f"{description}\n\n{hint}".strip()
        break
    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Skill definitions (used by H5 cold-start)
# ─────────────────────────────────────────────────────────────────────────────

ALF_SKILLS: List[Dict[str, Any]] = [
    {
        "id": "pickup_then_deliver",
        "task_types": ["pick_and_place", "pick_two_obj"],
        "keywords": ["put", "place", "locate", "navigate", "deliver"],
        "text": (
            "For placement tasks: locate the target object, take it, "
            "navigate directly to the destination receptacle, then put it there."
        ),
    },
    {
        "id": "state_changing_actions_only",
        "task_types": ["pick_clean_then_place", "pick_heat_then_place", "pick_cool_then_place"],
        "keywords": ["clean", "heat", "cool", "transform", "appliance"],
        "text": (
            "For transformation tasks: take the object to the required appliance "
            "(sinkbasin/microwave/fridge), apply the transformation, "
            "then deliver to the final destination."
        ),
    },
    {
        "id": "two_object_staging",
        "task_types": ["pick_two_obj"],
        "keywords": ["two", "both", "second", "pair", "another"],
        "text": (
            "You can carry one item at a time. Deliver the first object to the "
            "destination before collecting the second."
        ),
    },
    {
        "id": "examine_with_lamp",
        "task_types": ["look_at_obj"],
        "keywords": ["examine", "lamp", "light", "desklamp", "look"],
        "text": (
            "First go to the desklamp and turn it on, then find the target object, "
            "take it, and examine it with the desklamp."
        ),
    },
    {
        "id": "explore_unseen_first",
        "task_types": [
            "pick_and_place", "pick_two_obj",
            "pick_clean_then_place", "pick_heat_then_place", "pick_cool_then_place",
        ],
        "keywords": ["find", "locate", "search", "explore", "unvisited"],
        "text": (
            "When searching, prefer unvisited containers and locations over "
            "revisiting already-explored areas."
        ),
    },
    {
        "id": "break_loops_early",
        "task_types": [],   # not used for cold-start; injected by H4 recovery only
        "keywords": ["loop", "repeat", "stuck", "stall", "revisit"],
        "text": (
            "If recent actions revealed no new objects or progress, "
            "switch to a different unexplored location immediately."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat shim (used by task.py import)
# ─────────────────────────────────────────────────────────────────────────────

def first_sentence_query(user_prompt: str) -> str:
    """
    Legacy helper kept for import compatibility.
    No longer used by the harness itself (BM25 retrieval removed).
    """
    text = (user_prompt or "").strip()
    prefix = "Here is your task."
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    parts = re.split(r"(?<=[.!?])\s+|\n+", text, maxsplit=1)
    return parts[0].strip() if parts and parts[0].strip() else text
