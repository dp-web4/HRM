#!/usr/bin/env python3
"""
SAGE Resident Daemon for Thor

Always-on SAGE instance that raising sessions connect to.
Maintains model loaded in memory with persistent state.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configuration
DEFAULT_MODEL = "/home/dp/ai-workspace/HRM/model-zoo/sage/qwen2.5-7b-instruct"
DEFAULT_PORT = 8765
STATE_DIR = Path("/home/dp/ai-workspace/HRM/sage/raising/state")
IDENTITY_PATH = Path("/home/dp/ai-workspace/HRM/sage/raising/identity/IDENTITY.md")
HISTORY_PATH = Path("/home/dp/ai-workspace/HRM/sage/raising/identity/HISTORY.md")


class GenerateRequest(BaseModel):
    """Request for text generation"""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response from generation"""
    text: str
    tokens_generated: int
    inference_time_ms: float
    timestamp: str


class StateUpdate(BaseModel):
    """Update to SAGE's state"""
    conversation_history: Optional[List[Dict[str, str]]] = None
    memory_request: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SAGEServer:
    """
    Resident SAGE daemon - always-on instance with persistent state
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, port: int = DEFAULT_PORT):
        self.model_path = model_path
        self.port = port

        # Model components (loaded once)
        self.model = None
        self.tokenizer = None
        self.device = None

        # State (persists in memory between sessions)
        self.conversation_history: List[Dict[str, str]] = []
        self.identity = ""
        self.history = ""
        self.memory_request = ""
        self.metadata = {}

        # Statistics
        self.total_generations = 0
        self.uptime_start = time.time()
        self.last_activity = time.time()

        # API
        self.app = FastAPI(title="SAGE Resident Daemon")
        self._setup_routes()

        print(f"[SAGEServer] Initialized")
        print(f"  Model: {model_path}")
        print(f"  Port: {port}")
        print(f"  State dir: {STATE_DIR}")

    def _setup_routes(self):
        """Configure FastAPI routes"""

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "uptime_seconds": time.time() - self.uptime_start,
                "total_generations": self.total_generations,
                "last_activity": datetime.fromtimestamp(self.last_activity).isoformat()
            }

        @self.app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """Generate text from SAGE"""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            start_time = time.time()

            try:
                # Build full prompt with system prompt if provided
                full_prompt = request.prompt
                if request.system_prompt:
                    full_prompt = f"{request.system_prompt}\n\n{request.prompt}"

                # Tokenize
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.device)

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                # Update stats
                self.total_generations += 1
                self.last_activity = time.time()
                inference_time = (time.time() - start_time) * 1000

                return GenerateResponse(
                    text=generated_text,
                    tokens_generated=outputs.shape[1] - inputs['input_ids'].shape[1],
                    inference_time_ms=inference_time,
                    timestamp=datetime.now().isoformat()
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

        @self.app.get("/state")
        async def get_state():
            """Get current SAGE state"""
            return {
                "conversation_history": self.conversation_history[-10:],  # Last 10 exchanges
                "memory_request": self.memory_request,
                "metadata": self.metadata,
                "identity_loaded": len(self.identity) > 0,
                "history_loaded": len(self.history) > 0
            }

        @self.app.post("/state")
        async def update_state(update: StateUpdate):
            """Update SAGE state"""
            if update.conversation_history is not None:
                self.conversation_history = update.conversation_history

            if update.memory_request is not None:
                self.memory_request = update.memory_request

            if update.metadata is not None:
                self.metadata.update(update.metadata)

            self.last_activity = time.time()

            # Persist to disk
            await self.save_state()

            return {"status": "updated", "timestamp": datetime.now().isoformat()}

        @self.app.post("/reset")
        async def reset_state():
            """Reset conversation state (keeps identity)"""
            self.conversation_history = []
            self.memory_request = ""
            self.metadata = {}
            self.last_activity = time.time()

            return {"status": "reset", "timestamp": datetime.now().isoformat()}

    async def load_model(self):
        """Load model into memory (once, at startup)"""
        print(f"\n[SAGEServer] Loading model...")
        print(f"  Path: {self.model_path}")

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"  Device: CUDA (Thor unified memory)")

            # Check available memory
            props = torch.cuda.get_device_properties(0)
            total_mem = props.total_memory / 1024**3
            print(f"  Total memory: {total_mem:.1f} GB")
        else:
            self.device = "cpu"
            print(f"  Device: CPU")

        # Load tokenizer
        print("  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print("  Loading model (this may take a moment)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",  # Let transformers handle device mapping
            trust_remote_code=True
        )

        # Report memory usage
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory allocated: {allocated:.2f} GB")

        print("[SAGEServer] Model loaded successfully")

    async def load_identity(self):
        """Load SAGE identity and history"""
        print(f"\n[SAGEServer] Loading identity...")

        if IDENTITY_PATH.exists():
            self.identity = IDENTITY_PATH.read_text()
            print(f"  Identity loaded: {len(self.identity)} chars")
        else:
            print(f"  Warning: Identity file not found at {IDENTITY_PATH}")

        if HISTORY_PATH.exists():
            self.history = HISTORY_PATH.read_text()
            print(f"  History loaded: {len(self.history)} chars")
        else:
            print(f"  Warning: History file not found at {HISTORY_PATH}")

    async def load_state(self):
        """Load persisted state from disk"""
        state_file = STATE_DIR / "daemon_state.json"

        if state_file.exists():
            print(f"\n[SAGEServer] Loading persisted state...")
            with open(state_file) as f:
                state = json.load(f)

            self.conversation_history = state.get('conversation_history', [])
            self.memory_request = state.get('memory_request', '')
            self.metadata = state.get('metadata', {})

            print(f"  Conversation history: {len(self.conversation_history)} exchanges")
            print(f"  Memory request: {self.memory_request[:50]}..." if self.memory_request else "  No memory request")

    async def save_state(self):
        """Persist state to disk"""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_file = STATE_DIR / "daemon_state.json"

        state = {
            'conversation_history': self.conversation_history,
            'memory_request': self.memory_request,
            'metadata': self.metadata,
            'last_updated': datetime.now().isoformat()
        }

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    async def start(self):
        """Start the SAGE daemon"""
        print("\n" + "=" * 70)
        print("SAGE RESIDENT DAEMON - STARTING")
        print("=" * 70)

        # Load model (once)
        await self.load_model()

        # Load identity and history
        await self.load_identity()

        # Load persisted state
        await self.load_state()

        print("\n" + "=" * 70)
        print("SAGE RESIDENT DAEMON - READY")
        print("=" * 70)
        print(f"\nListening on http://localhost:{self.port}")
        print(f"Health check: http://localhost:{self.port}/health")
        print(f"\nPress Ctrl+C to shutdown")
        print("=" * 70 + "\n")

        # Start FastAPI server
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def shutdown(self):
        """Graceful shutdown"""
        print("\n[SAGEServer] Shutting down...")

        # Save state
        await self.save_state()
        print("  State saved")

        # Clear model from memory
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  Model unloaded")

        print("[SAGEServer] Shutdown complete")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="SAGE Resident Daemon")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    args = parser.parse_args()

    server = SAGEServer(model_path=args.model, port=args.port)

    try:
        await server.start()
    except KeyboardInterrupt:
        await server.shutdown()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
