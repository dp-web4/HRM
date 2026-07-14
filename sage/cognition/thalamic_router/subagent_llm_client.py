"""SubagentLLMClient — runs LLM calls via a Claude Code subagent loop.

When the LLM client is a Claude Code subagent (e.g., Sonnet via subscription),
there's no HTTP endpoint to POST to. Instead, the subagent acts as the LLM by
polling sentinel files: the sweep writes a prompt JSON to one file, blocks
until the subagent writes a response JSON to another file, then continues.

The subagent's prompt instructs it to loop:
  while not exists("done.flag"):
    if exists("pending.json"):
      reason about it
      write response to "response.json"

Each sweep instance uses a unique `name` so multiple subagents can run in
parallel without colliding.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class SubagentLLMClient:
    """LLM client that defers to a Claude Code subagent via sentinel files.

    Matches OllamaClient.chat() signature so play_with_plans can use it
    via llm_client= injection.
    """

    def __init__(
        self,
        name: str = "sage",
        model: str = "claude-sonnet-4-6",
        timeout: float = 600.0,
        sentinel_dir: str = "/tmp/subagent_llm",
    ):
        self.name = name
        self.model = model
        self.timeout = timeout
        self.call_times_s: List[float] = []

        self.dir = Path(sentinel_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.dir / "pending.json"
        self.response_path = self.dir / "response.json"
        self.done_path = self.dir / "done.flag"
        self.call_counter = 0

        # Clean any stale state from a prior run
        for p in [self.pending_path, self.response_path, self.done_path]:
            if p.exists():
                p.unlink()

    @property
    def model_name(self) -> str:
        return self.model

    def chat(
        self,
        prompt: str,
        images_png: Optional[List[bytes]] = None,
        max_tokens: int = 300,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Hand the prompt to the subagent, block on its response."""
        self.call_counter += 1
        request: Dict[str, Any] = {
            "call_id": self.call_counter,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        if history:
            request["history"] = history
        if system_prompt:
            request["system_prompt"] = system_prompt
        if images_png:
            # Write images as PNG files to disk so the subagent can Read them
            # natively (multimodal). Paths go in the request JSON.
            img_paths = []
            for i, img in enumerate(images_png):
                p = self.dir / f"frame_{self.call_counter:04d}_{i}.png"
                with open(p, "wb") as f:
                    f.write(img)
                img_paths.append(str(p))
            request["image_paths"] = img_paths

        # Atomic write: write to .tmp then rename so subagent never sees a
        # half-written file mid-poll
        tmp = self.pending_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(request, f)
        os.replace(tmp, self.pending_path)

        # Poll for response
        t0 = time.time()
        deadline = t0 + self.timeout
        while time.time() < deadline:
            if self.response_path.exists():
                try:
                    with open(self.response_path) as f:
                        resp = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Subagent might be mid-write. Try again.
                    time.sleep(0.2)
                    continue
                self.response_path.unlink()
                self.pending_path.unlink(missing_ok=True)
                self.call_times_s.append(time.time() - t0)
                return str(resp.get("text", ""))
            time.sleep(0.5)

        # Timeout
        self.call_times_s.append(time.time() - t0)
        return f"ERROR: SubagentLLMClient timeout after {self.timeout}s on call {self.call_counter}"

    def signal_done(self) -> None:
        """Tell the subagent the sweep is finished."""
        self.done_path.touch()
