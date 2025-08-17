
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone
from uuid import uuid4

# Pull in the minimal Totality implementation
from totality_min.totality_core import TotalityStore, Scheme, Slot

app = FastAPI(title="SAGE ↔ Totality Service (Integrated)", version="0.1")

T = TotalityStore()  # in-memory minimal Totality

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --------------------
# Pydantic DTOs for requests
# --------------------
class ReadFilters(BaseModel):
    activation_min: float = 0.0

class ImagineOp(BaseModel):
    op: str
    params: Optional[Dict[str, Any]] = None

class ImagineBody(BaseModel):
    seed_scheme_ids: List[str] = Field(default_factory=list)
    ops: List[ImagineOp] = Field(default_factory=list)
    count: int = 1

class ActivateTarget(BaseModel):
    id: str
    delta: float

class ActivateBody(BaseModel):
    targets: List[ActivateTarget]
    decay_half_life_s: Optional[float] = None

class SlotDTO(BaseModel):
    name: str
    value: Any = None
    binding: str = "observed"

class SchemeDTO(BaseModel):
    id: Optional[str] = None
    label: str
    slots: List[SlotDTO] = Field(default_factory=list)
    activation: float = 0.0

class WriteBody(BaseModel):
    schemes: List[SchemeDTO] = Field(default_factory=list)
    mode: str = "merge"   # merge|append|replace

# --------------------
# Seed some data for quick testing
# --------------------
def _seed():
    s1 = T.add_scheme("grasp(object)", slots=[{"name":"object","value":"cube"}], activation=0.6)
    s2 = T.add_scheme("place(object,location)", slots=[{"name":"object","value":"cube"},{"name":"location","value":"table"}], activation=0.4)
    T.add_canvas(description="base scene", entities=[s1, s2])
_seed()

# --------------------
# Endpoints
# --------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": now_iso()}

@app.get("/totality/read")
def read_totality(activation_min: float = Query(0.0, ge=0.0, le=1.0)):
    data = T.read(activation_min=activation_min)
    # Return JSONable dict from store helper
    return T.to_jsonable()

@app.post("/totality/imagine")
def imagine(body: ImagineBody):
    ops = [{"op": o.op, "params": o.params} for o in body.ops]
    canvases = T.imagine_from_schemes(body.seed_scheme_ids, ops=ops, count=body.count)
    return {"created": [c.id for c in canvases]}

@app.post("/totality/activate")
def activate(body: ActivateBody):
    T.activate([{"id": t.id, "delta": t.delta} for t in body.targets],
               decay_half_life_s=body.decay_half_life_s)
    return {"status": "ok", "activations": T.activations}

@app.post("/totality/write")
def write(body: WriteBody):
    schemes = []
    for sd in body.schemes:
        if sd.id is None:
            # create new ID
            sch = Scheme(id=str(uuid4()), label=sd.label)
        else:
            sch = Scheme(id=sd.id, label=sd.label)
        sch.activation = sd.activation
        sch.slots = [Slot(name=sl.name, value=sl.value, binding=sl.binding) for sl in sd.slots]
        schemes.append(sch)

    committed = T.write_schemes(schemes, mode=body.mode)
    return {"status": "ok", "committed": committed}

@app.get("/totality/snapshot")
def snapshot():
    snap = T.to_jsonable()
    snap_id = str(uuid4())
    return {"snapshot_id": snap_id, "snapshot": snap}
