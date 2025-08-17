
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from uuid import uuid4
from datetime import datetime, timezone

app = FastAPI(title="SAGE ↔ Totality Adapter (Stub)", version="0.1")

# ---------------------
# In-memory "Totality" store (stub)
# ---------------------
STORE = {
    "schemes": {},
    "viewpoints": {},
    "canvases": {},
    "activations": {},  # id -> float
    "snapshots": {},
}

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------------
# Pydantic models (subset of the spec for stub)
# ---------------------
class Confidence(BaseModel):
    p: float = 0.5

class Timestamp(BaseModel):
    iso8601: str = Field(default_factory=now_iso)

class Span(BaseModel):
    start: Timestamp
    end: Timestamp

class Affect(BaseModel):
    valence: float = 0.0   # [-1,1]
    arousal: float = 0.0   # [0,1]
    coping: Optional[str] = None
    tags: Optional[List[str]] = None

class TrustVector(BaseModel):
    source_reliability: float = 0.5
    context_alignment: float = 0.5
    recency_decay: float = 0.1
    affect_bias: float = 0.0
    composite: Optional[float] = None

class Slot(BaseModel):
    name: str
    value: Any = None
    binding: str = "observed"  # observed|imagined|inferred

class Scheme(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    slots: List[Slot] = []
    activation: float = 0.0
    support: Dict[str, Any] = {"evidence_ids": [], "confidence": {"p": 0.5}}

class ViewTransform(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = None

class Viewpoint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    frame: str = "world"
    transforms: List[ViewTransform] = []

class Link(BaseModel):
    src: str
    dst: str
    relation: str

class Canvas(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = ""
    entities: List[Scheme] = []
    links: List[Link] = []
    provenance: Dict[str, Any] = Field(default_factory=lambda: {
        "created_by": "Totality",
        "created_at": now_iso(),
        "ops": []
    })

class Summaries(BaseModel):
    text: str = ""
    keypoints: List[str] = []

class Features(BaseModel):
    schemes: List[Scheme] = []
    viewpoints: List[Viewpoint] = []
    canvases: List[Canvas] = []
    summaries: Summaries = Field(default_factory=Summaries)

class CognitiveReading(BaseModel):
    sensor_id: str = "totality"
    t: Timestamp = Field(default_factory=Timestamp)
    features: Features = Field(default_factory=Features)
    trust: TrustVector = Field(default_factory=TrustVector)
    affect: Affect = Field(default_factory=Affect)

# ---------------------
# Request models
# ---------------------
class ReadQuery(BaseModel):
    scope: List[str] = ["schemes","viewpoints","canvases"]
    filters: Dict[str, Any] = {}

class ImagineOp(BaseModel):
    op: str
    params: Optional[Dict[str, Any]] = None

class ImagineBody(BaseModel):
    seed: Dict[str, Any]
    ops: List[ImagineOp] = []
    affect_hint: Optional[Affect] = None
    count: int = 1

class ActivateTarget(BaseModel):
    id: str
    delta: float

class ActivateBody(BaseModel):
    targets: List[ActivateTarget]
    decay: Optional[Dict[str, Any]] = None

class WritePolicy(BaseModel):
    mode: str = "merge"     # merge|append|replace
    merge_keys: Optional[List[str]] = None

class WriteBody(BaseModel):
    schemes: List[Scheme] = []
    links: List[Link] = []
    policy: WritePolicy = Field(default_factory=WritePolicy)
    affect: Optional[Affect] = None

# ---------------------
# Helpers
# ---------------------
def _materialize(ids, pool):
    out = []
    for _id in ids:
        obj = pool.get(_id)
        if obj:
            out.append(obj)
    return out

def _compute_trust() -> TrustVector:
    tv = TrustVector()
    # naive composite formula (example)
    tv.composite = 0.4*tv.source_reliability + 0.3*tv.context_alignment + 0.2*(1 - tv.recency_decay) + 0.1*tv.affect_bias
    return tv

# ---------------------
# Endpoints
# ---------------------
@app.get("/totality/read", response_model=CognitiveReading)
def read_totality(scope: Optional[str] = None,
                  activation_min: float = 0.0):
    # Very simple filter by activation for demo purposes
    features = Features()
    if scope is None or "schemes" in scope:
        for s in STORE["schemes"].values():
            if s.activation >= activation_min:
                features.schemes.append(s)
    if scope is None or "viewpoints" in scope:
        features.viewpoints.extend(list(STORE["viewpoints"].values()))
    if scope is None or "canvases" in scope:
        features.canvases.extend(list(STORE["canvases"].values()))
    reading = CognitiveReading(features=features, trust=_compute_trust())
    return reading

@app.post("/totality/imagine", response_model=Dict[str, List[Canvas]])
def imagine(body: ImagineBody):
    # Stub: create 'count' canvases, optionally copying seed schemes
    canvases = []
    seed_ids = body.seed.get("ids", [])
    from_kind = body.seed.get("from", "schemes")
    seed_schemes = []
    if from_kind == "schemes":
        seed_schemes = _materialize(seed_ids, STORE["schemes"])
    for _ in range(body.count):
        c = Canvas(
            description=f"imagined from {from_kind}",
            entities=seed_schemes,
            provenance={
                "created_by": "SAGE",
                "created_at": now_iso(),
                "ops": [op.op for op in body.ops]
            }
        )
        STORE["canvases"][c.id] = c
        canvases.append(c)
    return {"canvases": canvases}

@app.post("/totality/activate", response_model=Dict[str, str])
def activate(body: ActivateBody):
    for tgt in body.targets:
        cur = STORE["activations"].get(tgt.id, 0.0)
        STORE["activations"][tgt.id] = max(0.0, min(1.0, cur + tgt.delta))
        # if scheme exists, mirror activation
        if tgt.id in STORE["schemes"]:
            s = STORE["schemes"][tgt.id]
            s.activation = STORE["activations"][tgt.id]
            STORE["schemes"][tgt.id] = s
    return {"status": "ok"}

@app.post("/totality/write", response_model=Dict[str, Any])
def write(body: WriteBody):
    committed = []
    # schemes
    for s in body.schemes:
        if body.policy.mode == "append" or s.id not in STORE["schemes"]:
            STORE["schemes"][s.id] = s
        elif body.policy.mode == "replace":
            STORE["schemes"][s.id] = s
        else:  # merge
            existing = STORE["schemes"].get(s.id)
            if existing:
                # naive merge: overwrite non-empty fields
                if s.label: existing.label = s.label
                if s.slots: existing.slots = s.slots
                existing.activation = max(existing.activation, s.activation)
                STORE["schemes"][s.id] = existing
            else:
                STORE["schemes"][s.id] = s
        committed.append(s.id)
    # links -> just attach to any canvases that reference src/dst (demo)
    # In a real impl, links belong to a graph store
    result = {"committed_ids": committed, "status": "ok"}
    return result

@app.get("/totality/snapshot", response_model=Dict[str, str])
def snapshot(include: Optional[str] = None):
    snap_id = str(uuid4())
    blob = {
        "time": now_iso(),
        "schemes": {k: v.dict() for k, v in STORE["schemes"].items()},
        "viewpoints": {k: v.dict() for k, v in STORE["viewpoints"].items()},
        "canvases": {k: v.dict() for k, v in STORE["canvases"].items()},
        "activations": STORE["activations"],
    }
    STORE["snapshots"][snap_id] = blob
    return {"snapshot_id": snap_id, "blob_uri": f"memory://{snap_id}"}

# ---------------------
# Seed some demo data for quick testing
# ---------------------
def _seed():
    s = Scheme(label="grasp(object)")
    s.activation = 0.6
    STORE["schemes"][s.id] = s
    v = Viewpoint()
    STORE["viewpoints"][v.id] = v
    c = Canvas(description="initial scene", entities=[s])
    STORE["canvases"][c.id] = c

_seed()
