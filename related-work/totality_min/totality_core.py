
# totality_core.py
# Minimal "Totality/SubThought" test implementation (no external deps)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid
import math
import copy

def uid() -> str:
    return str(uuid.uuid4())

@dataclass
class Slot:
    name: str
    value: Any = None
    binding: str = "observed"  # observed|imagined|inferred

@dataclass
class Scheme:
    id: str
    label: str
    slots: List[Slot] = field(default_factory=list)
    activation: float = 0.0
    support: Dict[str, Any] = field(default_factory=lambda: {"evidence_ids": [], "confidence": {"p": 0.5}})

    def clone(self) -> "Scheme":
        return copy.deepcopy(self)

@dataclass
class Link:
    src: str
    dst: str
    relation: str

@dataclass
class Canvas:
    id: str
    description: str = ""
    entities: List[Scheme] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=lambda: {
        "created_by": "Totality",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ops": []
    })

class TotalityStore:
    """A tiny in-memory world model with schemes, canvases, activations, and simple ops."""
    def __init__(self):
        self.schemes: Dict[str, Scheme] = {}
        self.canvases: Dict[str, Canvas] = {}
        self.activations: Dict[str, float] = {}  # id -> activation [0,1]

    # --- CRUD-ish operations ---
    def add_scheme(self, label: str, slots: Optional[List[Dict[str, Any]]] = None, activation: float = 0.0) -> Scheme:
        s = Scheme(id=uid(), label=label)
        if slots:
            s.slots = [Slot(**slot) for slot in slots]
        s.activation = activation
        self.schemes[s.id] = s
        self.activations[s.id] = activation
        return s

    def add_canvas(self, description: str, entities: Optional[List[Scheme]] = None, links: Optional[List[Dict[str, str]]] = None, provenance_ops: Optional[List[str]] = None) -> Canvas:
        c = Canvas(id=uid(), description=description)
        c.entities = [e.clone() for e in (entities or [])]
        if links:
            c.links = [Link(**lnk) for lnk in links]
        if provenance_ops:
            c.provenance["ops"] = list(provenance_ops)
        self.canvases[c.id] = c
        return c

    def read(self, activation_min: float = 0.0) -> Dict[str, Any]:
        schemes = [s for s in self.schemes.values() if s.activation >= activation_min]
        canvases = list(self.canvases.values())
        return {
            "schemes": schemes,
            "canvases": canvases
        }

    def activate(self, targets: List[Dict[str, Any]], decay_half_life_s: Optional[float] = None) -> None:
        for tgt in targets:
            _id = tgt["id"]
            delta = float(tgt.get("delta", 0.0))
            cur = float(self.activations.get(_id, 0.0))
            newv = max(0.0, min(1.0, cur + delta))
            self.activations[_id] = newv
            if _id in self.schemes:
                self.schemes[_id].activation = newv

    def write_schemes(self, schemes: List[Scheme], mode: str = "merge") -> List[str]:
        committed = []
        for s in schemes:
            if mode == "append" or s.id not in self.schemes:
                self.schemes[s.id] = s
            elif mode == "replace":
                self.schemes[s.id] = s
            else:  # merge
                existing = self.schemes.get(s.id)
                if existing:
                    if s.label: existing.label = s.label
                    if s.slots: existing.slots = s.slots
                    existing.activation = max(existing.activation, s.activation)
                    self.schemes[s.id] = existing
                else:
                    self.schemes[s.id] = s
            committed.append(s.id)
        return committed

    # --- Imagination / simple augmentation ---
    def imagine_from_schemes(self, scheme_ids: List[str], ops: List[Dict[str, Any]], count: int = 1) -> List[Canvas]:
        seeds = [self.schemes[sid] for sid in scheme_ids if sid in self.schemes]
        canvases = []
        for _ in range(max(1, count)):
            ents = []
            for s in seeds:
                sc = s.clone()
                # Apply very simple 'value permutations' (swap or shuffle slot values)
                slot_vals = [sl.value for sl in sc.slots]
                if slot_vals and len(slot_vals) > 1:
                    slot_vals = slot_vals[::-1]
                    for i, sl in enumerate(sc.slots):
                        sl.value = slot_vals[i]
                ents.append(sc)
            c = self.add_canvas(description="imagined canvas", entities=ents, provenance_ops=[op.get("op","?") for op in ops])
            canvases.append(c)
        return canvases

    # --- Utilities ---
    def to_jsonable(self) -> Dict[str, Any]:
        def scheme_dict(s: Scheme):
            return {
                "id": s.id, "label": s.label, "activation": s.activation,
                "slots": [{"name": sl.name, "value": sl.value, "binding": sl.binding} for sl in s.slots],
            }
        def canvas_dict(c: Canvas):
            return {
                "id": c.id, "description": c.description, "provenance": c.provenance,
                "entities": [scheme_dict(e) for e in c.entities],
                "links": [{"src": l.src, "dst": l.dst, "relation": l.relation} for l in c.links],
            }
        return {
            "schemes": [scheme_dict(s) for s in self.schemes.values()],
            "canvases": [canvas_dict(c) for c in self.canvases.values()],
            "activations": self.activations
        }
