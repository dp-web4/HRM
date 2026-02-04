# Policy Model Integration Guide

**Status**: Production-Ready (Session K Complete)
**Date**: 2026-02-04
**Version**: 1.0

This guide provides everything needed to integrate the policy interpreter into hardbound (TypeScript) and web4 (Python) projects.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Performance Metrics](#performance-metrics)
3. [Model Specifications](#model-specifications)
4. [Prompt Configuration](#prompt-configuration)
5. [Integration Patterns](#integration-patterns)
6. [Hardbound Integration](#hardbound-integration)
7. [Web4 Integration](#web4-integration)
8. [Monitoring & Observability](#monitoring--observability)
9. [Troubleshooting](#troubleshooting)
10. [Migration Path](#migration-path)

---

## Quick Start

### Prerequisites

- Python 3.8+ with llama-cpp-python
- Phi-4-mini 7B model (Q4_K_M GGUF format)
- 4GB+ GPU memory OR 8GB+ RAM (CPU fallback)

### 30-Second Integration

```python
from llama_cpp import Llama
from prompts_v4 import build_prompt_v4

# Load model (one-time, ~1.3s)
llm = Llama(
    model_path="/path/to/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,  # Auto-detect GPU
    verbose=False
)

# Analyze a situation
situation = {
    "action_type": "deploy",
    "actor_role": "developer",
    "actor_id": "user:alice",
    "t3_tensor": {"competence": 0.75, "reliability": 0.72, "integrity": 0.8},
    "resource": "env:production"
}

prompt = build_prompt_v4(situation, variant="hybrid")
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a policy interpreter."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=512,
    temperature=0.7
)

print(response['choices'][0]['message']['content'])
```

**Output**: Structured policy decision with reasoning.

---

## Performance Metrics

### Validated Quality (Session K)

| Metric | Value | Status |
|--------|-------|--------|
| **Pass Rate** | 100% (8/8 scenarios) | ✅ Production-ready |
| **Decision Accuracy** | 100% | ✅ Critical |
| **Reasoning Coverage** | 95.8% average | ✅ Excellent |
| **Model Loading** | ~1.3s | ✅ Fast |
| **Inference Latency** | ~2-4s per request | ✅ Acceptable |

### Prompt Variants

| Variant | Examples | Pass Rate | Coverage | Use Case |
|---------|----------|-----------|----------|----------|
| v3_condensed | 4 | 87-100% | 91-96% | High-efficiency |
| **v4_hybrid** | 5 | **100%** | **95.8%** | **Production (recommended)** |
| v2_fewshot | 8 | 100% | 95.8% | Legacy/verbose |

**Recommendation**: Use `v4_hybrid` for best reliability-efficiency balance.

### Computational Resources

**GPU (Recommended)**:
- VRAM: ~2.5GB (Q4_K_M quantization)
- Latency: ~2-3s per request
- Throughput: ~20-30 requests/minute

**CPU (Fallback)**:
- RAM: ~4GB
- Latency: ~10-15s per request
- Throughput: ~4-6 requests/minute

---

## Model Specifications

### Model Details

- **Name**: Phi-4-mini-instruct
- **Size**: 7B parameters (Q4_K_M: 2.49GB)
- **Context**: 131k tokens (using 8k window)
- **Format**: GGUF (llama.cpp compatible)

### Download

```bash
# Using Hugging Face CLI
huggingface-cli download microsoft/Phi-4-mini-instruct-gguf \
    microsoft_Phi-4-mini-instruct-Q4_K_M.gguf \
    --local-dir ./model-zoo/phi-4-mini-gguf
```

### Verification

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    n_ctx=8192,
    n_gpu_layers=-1
)

# Test inference
output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

assert output['choices'][0]['message']['content'], "Model loaded successfully"
```

---

## Prompt Configuration

### Recommended: v4_hybrid (5 examples)

```python
from prompts_v4 import build_prompt_v4

# Build prompt for your situation
prompt = build_prompt_v4(situation, variant="hybrid")
```

### Required Situation Fields

```python
situation = {
    # Required
    "action_type": str,        # "deploy", "commit", "read", etc.
    "actor_role": str,         # "developer", "admin", "ci_bot", etc.
    "actor_id": str,           # "user:alice", "bot:github-actions"
    "t3_tensor": {
        "competence": float,   # 0.0-1.0
        "reliability": float,  # 0.0-1.0
        "integrity": float     # 0.0-1.0
    },

    # Optional (but recommended)
    "resource": str,           # "env:production", "db:main"
    "team_context": str,       # "Production team with strict policies"
    "timestamp": str,          # ISO format
    "recent_history": str,     # "Alice has 100 successful deploys"
    "identity_metrics": dict   # {"level": "exemplary", "coherence": 0.98}
}
```

### Sampling Parameters

**Recommended settings**:
```python
{
    "temperature": 0.7,    # Balance creativity and consistency
    "top_p": 0.9,          # Nucleus sampling
    "max_tokens": 512,     # Sufficient for reasoning
}
```

**For deterministic output** (same situation → same decision):
```python
{
    "temperature": 0.0,    # Fully deterministic
    "max_tokens": 512,
}
```

---

## Integration Patterns

### Pattern 1: Synchronous Advisory

**Use case**: Interactive decisions where latency is acceptable.

```python
class PolicyAdvisor:
    def __init__(self, model_path: str):
        self.llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1)

    def get_advisory_opinion(self, situation: dict) -> dict:
        """Get policy decision with reasoning."""
        from prompts_v4 import build_prompt_v4

        prompt = build_prompt_v4(situation, variant="hybrid")

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a policy interpreter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7
        )

        return self.parse_response(response['choices'][0]['message']['content'])

    def parse_response(self, text: str) -> dict:
        """Extract decision and reasoning from response."""
        # Basic parsing (can be enhanced)
        decision = None
        if "decision: allow" in text.lower():
            decision = "allow"
        elif "decision: deny" in text.lower():
            decision = "deny"
        elif "decision: require_attestation" in text.lower():
            decision = "require_attestation"

        return {
            "decision": decision,
            "reasoning": text,
            "confidence": "high" if decision else "uncertain"
        }
```

### Pattern 2: Cached Pattern Library

**Use case**: Common situations that can be pre-computed.

```python
class CachedPolicyAdvisor:
    def __init__(self, model_path: str):
        self.llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1)
        self.pattern_cache = {}  # Hash → decision

    def get_decision(self, situation: dict) -> dict:
        """Get decision with caching for common patterns."""
        # Create cache key (hash situation)
        cache_key = self._hash_situation(situation)

        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]

        # Cache miss: query LLM
        decision = self.get_advisory_opinion(situation)

        # Cache for future
        if decision['confidence'] == 'high':
            self.pattern_cache[cache_key] = decision

        return decision

    def _hash_situation(self, situation: dict) -> str:
        """Create stable hash for caching."""
        import hashlib
        import json

        # Normalize for caching (remove timestamps, etc.)
        cacheable = {
            "action_type": situation.get("action_type"),
            "actor_role": situation.get("actor_role"),
            "resource": situation.get("resource"),
            "t3_avg": sum(situation.get("t3_tensor", {}).values()) / 3
        }

        return hashlib.sha256(json.dumps(cacheable, sort_keys=True).encode()).hexdigest()
```

### Pattern 3: Async Queue Processing

**Use case**: Background processing, batch decisions.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncPolicyAdvisor:
    def __init__(self, model_path: str):
        self.llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1)
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def get_decision_async(self, situation: dict) -> dict:
        """Async wrapper for LLM inference."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_advisory_opinion,
            situation
        )

    async def process_batch(self, situations: list) -> list:
        """Process multiple situations concurrently."""
        tasks = [self.get_decision_async(sit) for sit in situations]
        return await asyncio.gather(*tasks)
```

---

## Hardbound Integration

### Overview

Hardbound uses TypeScript with PolicyModel infrastructure. The LLM provides **advisory opinions** that inform but don't control the final decision engine.

### Architecture

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       │
       v
┌─────────────────┐
│  PolicyModel    │ (TypeScript - Hardbound)
│  Decision Engine│
└────────┬────────┘
         │
         ├──> Fast Path: Pattern Library (cached)
         │
         └──> Slow Path: LLM Advisory (Python/FFI)
                   │
                   v
              ┌──────────────┐
              │ Phi-4-mini   │
              │ Policy LLM   │
              └──────────────┘
```

### Integration via R6Request Adapter

From Session F, we have the R6Request adapter:

```typescript
// Hardbound (TypeScript)
import { PolicyModel } from './policy-model';

const policyModel = new PolicyModel();

// Convert hardbound request to R6Request format
const r6Request = {
    action_type: request.actionType,
    actor_role: actor.role,
    actor_id: actor.id,
    t3_tensor: {
        competence: actor.trust.competence,
        reliability: actor.trust.reliability,
        integrity: actor.trust.integrity
    },
    resource: request.resource,
    team_context: team.context
};

// Call Python LLM service (via HTTP/gRPC/FFI)
const advisory = await policyLLMService.getAdvisory(r6Request);

// PolicyModel makes final decision
const decision = policyModel.decide({
    request: r6Request,
    advisory: advisory,
    patterns: patternLibrary
});
```

### Deployment Options

**Option A: Sidecar Service** (Recommended)
```
Hardbound (TypeScript) → HTTP → Python LLM Service
```

**Option B: FFI/Native Module**
```
Hardbound (TypeScript) → Node FFI → Python Binding → LLM
```

**Option C: Message Queue**
```
Hardbound → Queue → LLM Worker → Response Queue
```

### Example Sidecar Service

```python
# llm_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from prompts_v4 import build_prompt_v4
from llama_cpp import Llama

app = FastAPI()

# Load model at startup
llm = Llama(
    model_path="/path/to/model.gguf",
    n_ctx=8192,
    n_gpu_layers=-1
)

class R6Request(BaseModel):
    action_type: str
    actor_role: str
    actor_id: str
    t3_tensor: dict
    resource: str = None
    team_context: str = None

@app.post("/advisory")
async def get_advisory(request: R6Request):
    """Get policy advisory opinion."""
    situation = request.dict()

    prompt = build_prompt_v4(situation, variant="hybrid")

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a policy interpreter."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7
    )

    return {
        "advisory": response['choices'][0]['message']['content'],
        "model": "phi-4-mini-7b",
        "version": "v4_hybrid"
    }

# Run: uvicorn llm_service:app --host 0.0.0.0 --port 8000
```

---

## Web4 Integration

### Overview

Web4 uses Python natively, so integration is more direct.

### Direct Integration

```python
# web4/policy.py
from llama_cpp import Llama
from prompts_v4 import build_prompt_v4

class Policy:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1
        )

    def evaluate(self, action: dict, actor: dict, resource: dict) -> dict:
        """Evaluate action against team policy."""
        # Build situation from web4 data structures
        situation = {
            "action_type": action['type'],
            "actor_role": actor['role'],
            "actor_id": actor['id'],
            "t3_tensor": actor['trust'],
            "resource": resource['id'],
            "team_context": action.get('team_context', '')
        }

        # Get LLM advisory
        prompt = build_prompt_v4(situation, variant="hybrid")

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a policy interpreter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7
        )

        advisory_text = response['choices'][0]['message']['content']

        # Parse decision
        decision = self._parse_decision(advisory_text)

        return {
            "decision": decision,
            "reasoning": advisory_text,
            "source": "llm_advisory"
        }

    def _parse_decision(self, text: str) -> str:
        """Extract decision from LLM response."""
        text_lower = text.lower()
        if "decision: allow" in text_lower:
            return "allow"
        elif "decision: deny" in text_lower:
            return "deny"
        elif "decision: require_attestation" in text_lower:
            return "require_attestation"
        return "escalate"  # Uncertain → human review
```

### Singleton Pattern (Recommended)

```python
# web4/policy_singleton.py
class PolicyAdvisorSingleton:
    _instance = None
    _llm = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path="/path/to/model.gguf",
                n_ctx=8192,
                n_gpu_layers=-1
            )

    def get_advisory(self, situation: dict) -> dict:
        from prompts_v4 import build_prompt_v4

        prompt = build_prompt_v4(situation, variant="hybrid")

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a policy interpreter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7
        )

        return {
            "decision": self._parse(response),
            "reasoning": response['choices'][0]['message']['content']
        }

# Usage
advisor = PolicyAdvisorSingleton()
decision = advisor.get_advisory(situation)
```

---

## Monitoring & Observability

### Key Metrics to Track

1. **Latency**
   - Model load time (one-time: ~1.3s)
   - Inference time per request (target: <5s)
   - End-to-end advisory time

2. **Quality**
   - Decision distribution (allow/deny/attestation)
   - Human override rate (advisory vs final decision)
   - Reasoning coverage (if evaluating)

3. **Resource Usage**
   - GPU/CPU utilization
   - Memory consumption
   - Request queue depth

### Logging Example

```python
import logging
import time

logger = logging.getLogger("policy_llm")

class MonitoredPolicyAdvisor:
    def __init__(self, model_path: str):
        self.llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1)
        self.stats = {
            "total_requests": 0,
            "decisions": {"allow": 0, "deny": 0, "require_attestation": 0},
            "avg_latency": 0.0
        }

    def get_advisory(self, situation: dict) -> dict:
        start = time.time()

        try:
            # Inference
            prompt = build_prompt_v4(situation, variant="hybrid")
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a policy interpreter."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.7
            )

            decision_text = response['choices'][0]['message']['content']
            decision = self._parse_decision(decision_text)

            # Update stats
            latency = time.time() - start
            self.stats["total_requests"] += 1
            self.stats["decisions"][decision] = self.stats["decisions"].get(decision, 0) + 1
            self.stats["avg_latency"] = (
                (self.stats["avg_latency"] * (self.stats["total_requests"] - 1) + latency)
                / self.stats["total_requests"]
            )

            logger.info(f"Advisory: {decision} ({latency:.2f}s)")

            return {"decision": decision, "reasoning": decision_text}

        except Exception as e:
            logger.error(f"Advisory failed: {e}")
            return {"decision": "escalate", "reasoning": f"Error: {e}"}
```

---

## Troubleshooting

### Common Issues

#### Model fails to load

**Symptoms**: `OSError: /path/to/model.gguf not found`

**Solution**:
```bash
# Verify file exists
ls -lh /path/to/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf

# Check permissions
chmod 644 /path/to/model.gguf
```

#### GPU out of memory

**Symptoms**: `CUDA error: out of memory`

**Solution**:
```python
# Reduce GPU layers (hybrid CPU/GPU)
llm = Llama(
    model_path="/path/to/model.gguf",
    n_ctx=8192,
    n_gpu_layers=20,  # Use fewer GPU layers
)

# OR: Full CPU fallback
llm = Llama(
    model_path="/path/to/model.gguf",
    n_ctx=8192,
    n_gpu_layers=0,  # CPU only
)
```

#### Slow inference (~20s+)

**Symptoms**: Inference takes much longer than expected

**Solutions**:
1. Check GPU is being used: `n_gpu_layers=-1`
2. Verify CUDA availability: `nvidia-smi`
3. Reduce context window: `n_ctx=4096`
4. Use GPU if on CPU

#### Inconsistent decisions

**Symptoms**: Same situation gives different decisions

**Cause**: `temperature > 0` introduces sampling variance

**Solution**:
```python
# For deterministic output
response = llm.create_chat_completion(
    messages=[...],
    temperature=0.0,  # Deterministic
    max_tokens=512
)
```

---

## Migration Path

### From No Policy LLM → Initial Deployment

**Week 1: Setup**
- Deploy LLM service (sidecar/direct integration)
- Configure model loading
- Test basic inference

**Week 2: Shadow Mode**
- Run LLM alongside existing policy engine
- Log advisory opinions
- Compare with current decisions
- Measure latency and quality

**Week 3: Advisory Mode**
- Surface LLM reasoning to human reviewers
- Use for complex/edge cases only
- Collect feedback

**Week 4+: Gradual Rollout**
- Start with low-risk actions (reads)
- Expand to moderate-risk (writes, commits)
- High-risk (deploys, admin) requires human review
- Monitor override rates

### Success Criteria

- ✅ Advisory latency <5s (p95)
- ✅ Human override rate <10% (advisory matches human judgment)
- ✅ Zero critical errors (wrong allow on high-risk action)
- ✅ Model uptime >99%

---

## Summary

### Production-Ready Components

- ✅ **Prompts**: v4_hybrid (5 examples, 100% pass rate)
- ✅ **Model**: Phi-4-mini 7B Q4_K_M (2.49GB, fast inference)
- ✅ **Evaluation**: Validated with 95.8% reasoning coverage
- ✅ **Integration patterns**: Sync, async, cached options

### Next Steps

1. **Choose integration pattern** (sidecar recommended for hardbound)
2. **Deploy model** (GPU recommended, CPU fallback works)
3. **Start in shadow mode** (log advisories, don't enforce)
4. **Measure and iterate** (latency, quality, overrides)

### Support

- Session summaries: `SESSION_SUMMARY_*.md`
- Test suite: `test_suite_semantic.py`
- Prompts: `prompts_v4.py` (recommended), `prompts_v3.py`, `prompts_v2.py`
- Logging: `policy_logging.py`

---

**Version**: 1.0
**Status**: Production-Ready
**Last Updated**: 2026-02-04 (Session K)
**Maintained by**: Thor Policy Training Track
