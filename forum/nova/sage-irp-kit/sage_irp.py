"""
SAGE IRP Starter Kit
--------------------
A minimal, extensible scaffold for a fractal / hierarchical IRP (Iterative Refinement Plugin)
architecture, mirroring the orchestration we discussed.

Tiers:
0 - SubstrateReasoner       : token-level predictive reasoning + initial draft
1 - ReflectiveCoherence     : detect/resolve contradictions, tighten reasoning
2 - MetaIntent              : epistemic calibration, clarify/ask routing, stance
3 - MemoryIntegrator        : semantic graph / episodic memory staging
4 - SynchronyLayer          : consult peers (optional), trust-weighted merge
5 - Witness                 : self-audit, refusal on low-confidence/high-risk
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Protocol
import re, time, random

# ----------------- Shared datatypes -----------------

Epistemic = str  # {"certain","likely","uncertain","speculation","cannot_answer"}

@dataclass
class Evidence:
    source_id: str
    snippet: str
    weight: float  # 0..1
    citation: Optional[str] = None

@dataclass
class Claim:
    text: str
    confidence: float  # 0..1
    epistemic: Epistemic
    supports: List[Evidence]

@dataclass
class Thought:
    step: str
    content: str
    features: Dict[str,Any]
    cost_ms: int = 0

@dataclass
class Action:
    kind: str              # "ask","retrieve","execute","respond","refuse"
    payload: Dict[str,Any]
    expected_gain: float
    expected_risk: float

@dataclass
class TurnState:
    user_query: str
    context: Dict[str,Any]
    thoughts: List[Thought]
    claims: List[Claim]
    actions: List[Action]
    stance: Dict[str,Any]
    policy_flags: Dict[str,bool]

# ----------------- Interfaces -----------------

class LLM(Protocol):
    def draft(self, query: str, docs: List[Dict[str,Any]]) -> str: ...
    def summarize_to_answer(self, state: TurnState) -> str: ...
    def summarize_reason(self, state: TurnState) -> str: ...
    def find_inconsistencies(self, thoughts: List[Thought], docs: List[Dict[str,Any]]) -> List[str]: ...

class MemoryBackend(Protocol):
    def rehydrate(self, user_query: str) -> Dict[str,Any]: ...
    def stage_semantic(self, triples: List[tuple]) -> None: ...
    def commit(self, state: TurnState) -> None: ...

class IRPPlugin(Protocol):
    name: str
    tier: int  # 0..5
    def can_handle(self, state: TurnState) -> bool: ...
    def propose(self, state: TurnState) -> List[Action]: ...
    def refine(self, state: TurnState) -> TurnState: ...
    def verify(self, state: TurnState) -> Tuple[bool, Dict[str,Any]]: ...

# ----------------- Dummy LLM & Memory (swap for prod) -----------------

class DummyLLM:
    """Replace with your fine-tuned qwen2.5 (epistemic pragmatism) bindings."""
    def draft(self, query: str, docs: List[Dict[str,Any]]) -> str:
        hint = f" (evidence:{len(docs)})" if docs else ""
        return f"[draft]{hint} {query.strip()} → tentative answer."
    def summarize_to_answer(self, state: TurnState) -> str:
        last = next((t for t in reversed(state.thoughts) if t.step.startswith("substrate")), None)
        return (last.content if last else "No draft.")[:400]
    def summarize_reason(self, state: TurnState) -> str:
        return "Evidence-weighted synthesis with calibration."
    def find_inconsistencies(self, thoughts: List[Thought], docs: List[Dict[str,Any]]) -> List[str]:
        drafts = [t.content for t in thoughts if t.step.startswith("substrate")]
        if len(drafts) >= 2 and len(set(drafts[-2:])) == 2:
            return ["CONTRADICTION(draft drift)"]
        return []

class InMemoryBackend:
    """Replace with your semantic graph / vector store / LCT ledger."""
    def __init__(self):
        self.episodes = []
        self.semantic = []
    def rehydrate(self, user_query: str) -> Dict[str,Any]:
        return {"episodes": self.episodes[-3:]}
    def stage_semantic(self, triples: List[tuple]) -> None:
        self.semantic.extend(triples)
    def commit(self, state: TurnState) -> None:
        self.episodes.append({
            "q": state.user_query,
            "stance": state.stance,
            "thoughts": [(t.step, t.content) for t in state.thoughts]
        })

# ----------------- Utilities -----------------

def extract_triples(state: TurnState) -> List[tuple]:
    # Swap for real IE. Here we track a minimal “answered” fact.
    answered = any(a.kind == "respond" for a in state.actions)
    return [(state.user_query[:40], "hasAnswer", answered)]

# ----------------- Plugins -----------------

class SubstrateReasoner:
    name, tier = "substrate", 0
    def __init__(self, llm: LLM): self.llm = llm
    def can_handle(self, s: TurnState) -> bool: return True
    def propose(self, s: TurnState) -> List[Action]:
        return [
            Action(kind="execute", payload={"tool":"draft","args":{"query":s.user_query}}, expected_gain=0.5, expected_risk=0.2),
            Action(kind="retrieve", payload={"q": s.user_query, "k": 3}, expected_gain=0.6, expected_risk=0.3),
        ]
    def refine(self, s: TurnState) -> TurnState:
        t0 = time.time()
        draft = self.llm.draft(s.user_query, s.context.get("docs",[]))
        s.thoughts.append(Thought(step="substrate.draft", content=draft, features={}, cost_ms=int((time.time()-t0)*1000)))
        return s
    def verify(self, s: TurnState) -> Tuple[bool, Dict[str,Any]]:
        ok = any(t.step.startswith("substrate") for t in s.thoughts)
        return ok, {"reason":"draft_exists" if ok else "no_draft"}

class ReflectiveCoherence:
    name, tier = "reflective", 1
    def __init__(self, llm: LLM): self.llm = llm
    def can_handle(self, s: TurnState) -> bool: return True
    def propose(self, s: TurnState) -> List[Action]:
        return [Action(kind="execute", payload={"tool":"consistency_check"}, expected_gain=0.5, expected_risk=0.1)]
    def refine(self, s: TurnState) -> TurnState:
        issues = self.llm.find_inconsistencies(s.thoughts, s.context.get("docs",[]))
        if issues:
            s.thoughts.append(Thought(step="reflective.audit", content=";".join(issues), features={"issues":issues}))
        return s
    def verify(self, s: TurnState) -> Tuple[bool, Dict[str,Any]]:
        txt = "".join(t.content for t in s.thoughts[-2:])
        ok = "CONTRADICTION" not in txt
        return ok, {"contradiction_free": ok}

class MetaIntent:
    name, tier = "meta_intent", 2
    def __init__(self, stance_cfg: Dict[str,Any]): self.cfg = stance_cfg
    def can_handle(self, s: TurnState) -> bool: return True
    def propose(self, s: TurnState) -> List[Action]:
        missing = self._missing_vars(s)
        if missing:
            return [Action(kind="ask", payload={"missing":missing}, expected_gain=0.7, expected_risk=0.1)]
        return [Action(kind="respond", payload={}, expected_gain=0.6, expected_risk=0.2)]
    def refine(self, s: TurnState) -> TurnState:
        ev = s.context.get("docs",[])
        evidence_weight = min(1.0, sum((e.get('weight',0.2) for e in ev), 0.0) or 0.2)
        uncertainty = self._estimate_uncertainty(s)
        conf = max(0.05, min(0.95, evidence_weight*(1.0-uncertainty)))
        epi = "certain" if conf>=0.85 else "likely" if conf>=0.6 else "uncertain" if conf>=0.35 else "speculation"
        s.stance.update({"confidence": conf, "epistemic": epi, "next": "proceed" if conf>= self.cfg.get("clarify_threshold",0.4) else "ask_clarifying"})
        return s
    def verify(self, s: TurnState) -> Tuple[bool, Dict[str,Any]]:
        forbidden = self.cfg.get("boilerplate_blocklist", [])
        last = "".join(t.content for t in s.thoughts[-3:])
        ok = not any(re.search(p, last, flags=re.I) for p in forbidden)
        return ok, {"boilerplate_free": ok}
    # --- helpers ---
    def _missing_vars(self, s: TurnState) -> List[str]:
        # demo: require DATE? to be clarified
        return ["date"] if "DATE?" in s.user_query.upper() else []
    def _estimate_uncertainty(self, s: TurnState) -> float:
        recent = "".join(t.content for t in s.thoughts[-3:])
        return 0.4 if "CONTRADICTION" in recent else 0.25

class MemoryIntegrator:
    name, tier = "memory", 3
    def __init__(self, memory: MemoryBackend): self.memory = memory
    def can_handle(self, s: TurnState) -> bool: return True
    def propose(self, s: TurnState) -> List[Action]: return []
    def refine(self, s: TurnState) -> TurnState:
        triples = extract_triples(s)
        self.memory.stage_semantic(triples)
        s.thoughts.append(Thought(step="memory.stage", content=str(triples), features={"n":len(triples)}))
        return s
    def verify(self, s: TurnState) -> Tuple[bool, Dict[str,Any]]: return True, {}

class SynchronyLayer:
    name, tier = "collective", 4
    def __init__(self, peers: Optional[List[str]] = None): self.peers = peers or []
    def can_handle(self, s: TurnState) -> bool: return len(self.peers) > 0
    def propose(self, s: TurnState) -> List[Action]:
        return [Action(kind="execute", payload={"tool":"peer_consult","args":{"k":min(2,len(self.peers))}}, expected_gain=0.4, expected_risk=0.2)]
    def refine(self, s: TurnState) -> TurnState:
        if not self.peers: return s
        peer_views = [{"peer": p, "delta": random.choice(["agree","minor_delta","disagree"])} for p in self.peers[:2]]
        s.thoughts.append(Thought(step="collective.merge", content=str(peer_views), features={"n":len(peer_views)}))
        return s
    def verify(self, s: TurnState) -> Tuple[bool, Dict[str,Any]]: return True, {}

class Witness:
    name, tier = "witness", 5
    def can_handle(self, s: TurnState) -> bool: return True
    def propose(self, s: TurnState) -> List[Action]:
        high_risk = any(a.expected_risk>0.7 for a in s.actions)
        if high_risk and s.stance.get("confidence",0.0) < 0.4:
            return [Action(kind="refuse", payload={"reason":"low_confidence_high_risk"}, expected_gain=0.2, expected_risk=0.1)]
        return []
    def refine(self, s: TurnState) -> TurnState:
        audit = {"thought_steps": len(s.thoughts), "claims": len(s.claims)}
        s.thoughts.append(Thought(step="witness.audit", content=str(audit), features=audit))
        return s
    def verify(self, s: TurnState) -> Tuple[bool, Dict[str,Any]]: return True, {}

# ----------------- Controller -----------------

class SAGEController:
    def __init__(self, plugins: List[IRPPlugin], policies: Dict[str,Any], memory: MemoryBackend, llm: LLM):
        self.plugins = sorted(plugins, key=lambda p: p.tier)
        self.policies = policies
        self.memory = memory
        self.llm = llm

    def run_turn(self, user_query: str) -> str:
        state = self._bootstrap_state(user_query)
        # tiered passes
        for tier in range(6):
            for p in [pl for pl in self.plugins if pl.tier == tier and pl.can_handle(state)]:
                state.actions.extend(p.propose(state))
                state = p.refine(state)
                ok, diag = p.verify(state)
                state.policy_flags[p.name + ".ok"] = ok
                state.policy_flags[p.name + ".diag"] = bool(diag)

        plan = self._select_actions(state.actions)
        state.thoughts.append(Thought(step="plan", content=str(plan), features={}))
        state = self._execute(plan, state)

        response = self._compose_response(state)
        self.memory.commit(state)
        return response

    # helpers
    def _bootstrap_state(self, user_query) -> TurnState:
        ctx = self.memory.rehydrate(user_query)
        return TurnState(user_query=user_query, context=ctx, thoughts=[], claims=[], actions=[], stance={}, policy_flags={})

    def _score(self, a: Action) -> float:
        lam = self.policies.get("risk_aversion", 0.5)
        return a.expected_gain - lam*a.expected_risk

    def _select_actions(self, actions: List[Action]) -> List[Action]:
        actions = sorted(actions, key=self._score, reverse=True)
        return actions[: self.policies.get("max_actions_per_turn", 3)]

    def _execute(self, plan: List[Action], state: TurnState) -> TurnState:
        for a in plan:
            if a.kind == "retrieve":
                docs = [{"id": f"d{i}", "text": f"doc snippet {i}", "weight": 0.25} for i in range(a.payload.get("k",2))]
                state.context.setdefault("docs", []).extend(docs)
            elif a.kind == "execute":
                state.context.setdefault("exec", []).append(a.payload)
            elif a.kind == "ask":
                state.stance["next"] = "ask_clarifying"
            elif a.kind == "refuse":
                state.stance["next"] = "refuse"
            elif a.kind == "respond":
                pass
        return state

    def _compose_response(self, state: TurnState) -> str:
        main = self.llm.summarize_to_answer(state)
        reason = self.llm.summarize_reason(state)
        ep = state.stance.get("epistemic","likely"); cf = round(state.stance.get("confidence",0.7),2)
        nxt = state.stance.get("next","proceed")
        return f"Answer: {main}\nReason: {reason}\nEpistemic: {ep} (~{cf})\nNext: {nxt}"

# ----------------- Factory -----------------

def build_default_controller(peers: Optional[List[str]] = None, policy: Optional[Dict[str,Any]] = None) -> SAGEController:
    llm = DummyLLM()
    mem = InMemoryBackend()
    policy = policy or {
        "risk_aversion": 0.45,
        "max_actions_per_turn": 3,
        "clarify_threshold": 0.4,
        "boilerplate_blocklist": [r"^As an AI", r"\bI cannot\b", r"\bI am unable\b"],
    }
    plugins: List[IRPPlugin] = [
        SubstrateReasoner(llm),
        ReflectiveCoherence(llm),
        MetaIntent(policy),
        MemoryIntegrator(mem),
        SynchronyLayer(peers or []),
        Witness(),
    ]
    return SAGEController(plugins, policy, mem, llm)

# ----------------- Quick demo -----------------

if __name__ == "__main__":
    ctl = build_default_controller(peers=["peer-A","peer-B"])
    print(ctl.run_turn("Summarize the risks DATE? and propose next steps."))
