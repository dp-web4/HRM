# plugs your fine-tuned model into the SAGE IRP controller
from typing import Optional, Dict, Any, List
from sage_irp import build_default_controller, SAGEController

def make_controller(peers: Optional[List[str]] = None,
                    policy: Optional[Dict[str,Any]] = None) -> SAGEController:
    policy = policy or {
        "risk_aversion": 0.45,
        "max_actions_per_turn": 3,
        "clarify_threshold": 0.4,
        "boilerplate_blocklist": [r"^As an AI", r"\bI cannot\b", r"\bI am unable\b"],
    }
    # swap build_default_controller with your variant that binds qwen2.5-0.5b-epiprag
    return build_default_controller(peers=peers or [], policy=policy)
