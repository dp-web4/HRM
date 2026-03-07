#!/usr/bin/env python3
"""
Thor Qwen3.5-27B Instance Daemon Starter

Loads the thor-qwen3.5-27b instance and starts the SAGE daemon.
"""

import sys
import json
from pathlib import Path

# Add SAGE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sage.irp.plugins.qwen35_27b_lora_irp import Qwen35_27B_LoRA_IRP
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from typing import Dict, Any, Optional, List


# Load instance configuration
INSTANCE_DIR = Path(__file__).parent
CONFIG_FILE = INSTANCE_DIR / 'instance.json'

with open(CONFIG_FILE) as f:
    INSTANCE_CONFIG = json.load(f)

print(f"Loading instance: {INSTANCE_CONFIG['slug']}")
print(f"  Machine: {INSTANCE_CONFIG['machine']}")
print(f"  Model: {INSTANCE_CONFIG['model']}")
print(f"  Backend: {INSTANCE_CONFIG['backend']}")
print(f"  Training: {INSTANCE_CONFIG['has_training_track']}")
print(f"  LoRA: {INSTANCE_CONFIG['has_lora']}")


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    memory: Optional[List[Dict[str, str]]] = None
    images: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    response: str
    iterations: int
    final_energy: float
    trust_score: float


class StatusResponse(BaseModel):
    instance: str
    model: str
    status: str
    training_enabled: bool
    active_adapter: Optional[str]
    experiences_collected: int
    trust_score: float


# Initialize FastAPI
app = FastAPI(
    title=f"SAGE Daemon - {INSTANCE_CONFIG['slug']}",
    description=f"Thor Qwen3.5-27B with LoRA training",
    version="0.1.0"
)

# Global IRP plugin instance
irp_plugin = None


@app.on_event("startup")
async def startup():
    """Load model on startup"""
    global irp_plugin

    print("\n" + "=" * 80)
    print("SAGE DAEMON - Thor Qwen3.5-27B Instance")
    print("=" * 80)

    # Configure IRP plugin from instance config
    plugin_config = {
        'model_path': INSTANCE_CONFIG['model_path'],
        'instance_dir': str(INSTANCE_DIR),
        'training_enabled': INSTANCE_CONFIG['has_lora'],
        'multimodal_enabled': INSTANCE_CONFIG.get('multimodal_enabled', True),
        'lora_r': INSTANCE_CONFIG['lora_config']['r'],
        'lora_alpha': INSTANCE_CONFIG['lora_config']['lora_alpha'],
        'lora_dropout': INSTANCE_CONFIG['lora_config']['lora_dropout'],
        'lora_target_modules': INSTANCE_CONFIG['lora_config']['target_modules'],
        'consolidation_threshold': INSTANCE_CONFIG['sleep_cycle_config']['consolidation_threshold'],
    }

    print("\nInitializing IRP plugin...")
    irp_plugin = Qwen35_27B_LoRA_IRP(config=plugin_config)

    print("\n" + "=" * 80)
    print("SAGE DAEMON - READY")
    print("=" * 80)
    print(f"\nEndpoints:")
    print(f"  POST /generate - Generate text")
    print(f"  GET  /status   - Instance status")
    print(f"  GET  /health   - Health check")
    print("=" * 80 + "\n")


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "instance": INSTANCE_CONFIG['slug'],
        "model_loaded": irp_plugin is not None,
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get instance status"""
    if irp_plugin is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    status = irp_plugin.get_status()

    return StatusResponse(
        instance=INSTANCE_CONFIG['slug'],
        model=status['model'],
        status="ready",
        training_enabled=status['training_enabled'],
        active_adapter=status['active_adapter'],
        experiences_collected=status['experiences_collected'],
        trust_score=status['trust_score'],
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using IRP protocol"""
    if irp_plugin is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare context
    context = {
        'prompt': request.prompt,
        'memory': request.memory or [],
        'images': request.images or [],
    }

    # IRP protocol: init → refine until converged → finalize
    state = irp_plugin.init_state(context)

    while not irp_plugin.converged(state):
        state = irp_plugin.refine(state)

    result = irp_plugin.finalize(state)

    return GenerateResponse(
        response=result['response'],
        iterations=result['iterations'],
        final_energy=result['final_energy'],
        trust_score=result['trust_score'],
    )


@app.post("/sleep_cycle")
async def trigger_sleep_cycle():
    """Manually trigger sleep cycle consolidation"""
    if irp_plugin is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not irp_plugin.training_enabled:
        raise HTTPException(status_code=400, detail="Training not enabled")

    experiences_count = len(irp_plugin.experience_buffer)

    # TODO: Implement actual sleep cycle training
    # For now, just report status

    return {
        "status": "sleep_cycle_queued",
        "experiences_pending": experiences_count,
        "message": "Sleep cycle training not yet implemented"
    }


def main():
    """Start the daemon"""
    import argparse

    parser = argparse.ArgumentParser(description="Thor Qwen3.5-27B SAGE Daemon")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
