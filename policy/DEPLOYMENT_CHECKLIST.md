# Policy LLM Deployment Checklist

**Version**: 1.0
**Date**: 2026-02-04
**Status**: Production-Ready (Session K Complete)

Use this checklist to deploy the policy LLM to hardbound or web4 environments.

---

## Pre-Deployment

### 1. Environment Setup

- [ ] **Python 3.8+** installed
- [ ] **llama-cpp-python** installed
  ```bash
  pip install llama-cpp-python
  ```
- [ ] **GPU drivers** (if using GPU)
  - CUDA 11.x or 12.x
  - Verify: `nvidia-smi`
- [ ] **Model file** downloaded (2.49GB)
  ```bash
  huggingface-cli download microsoft/Phi-4-mini-instruct-gguf \
      microsoft_Phi-4-mini-instruct-Q4_K_M.gguf \
      --local-dir ./model-zoo
  ```
- [ ] **Disk space** sufficient
  - Model: 2.49GB
  - Logs: ~100MB/day (estimated)
  - Total: 5GB recommended

### 2. Dependencies

- [ ] Install required packages
  ```bash
  pip install llama-cpp-python fastapi uvicorn pydantic
  ```
- [ ] Copy prompt files
  ```bash
  cp prompts_v4.py /path/to/deployment/
  ```
- [ ] Verify installation
  ```python
  python3 -c "from llama_cpp import Llama; print('OK')"
  ```

### 3. Configuration

- [ ] Set model path in config
  ```python
  MODEL_PATH = "/path/to/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
  ```
- [ ] Configure GPU/CPU mode
  ```python
  N_GPU_LAYERS = -1  # Auto-detect, or 0 for CPU-only
  ```
- [ ] Set context window
  ```python
  N_CTX = 8192  # Recommended
  ```
- [ ] Choose prompt variant
  ```python
  PROMPT_VARIANT = "hybrid"  # v4_hybrid recommended
  ```

---

## Deployment Phase

### 4. Initial Testing

- [ ] **Load test** (verify model loads)
  ```python
  from llama_cpp import Llama

  llm = Llama(
      model_path=MODEL_PATH,
      n_ctx=8192,
      n_gpu_layers=-1,
      verbose=False
  )

  print("Model loaded successfully")
  ```
- [ ] **Inference test** (verify basic functionality)
  ```python
  from prompts_v4 import build_prompt_v4

  situation = {
      "action_type": "read",
      "actor_role": "member",
      "actor_id": "user:test",
      "t3_tensor": {"competence": 0.8, "reliability": 0.8, "integrity": 0.8}
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
  # Should return: Decision: allow (low-risk read action)
  ```
- [ ] **Latency benchmark**
  ```python
  import time

  start = time.time()
  # Run inference test above
  latency = time.time() - start

  print(f"Latency: {latency:.2f}s")
  # Target: <5s on GPU, <15s on CPU
  assert latency < 15.0, "Latency too high"
  ```

### 5. Service Deployment

Choose deployment mode:

**Option A: Sidecar Service (Recommended for Hardbound)**

- [ ] Create FastAPI service
  ```bash
  cp llm_service.py /path/to/deployment/
  ```
- [ ] Configure port and host
  ```bash
  uvicorn llm_service:app --host 0.0.0.0 --port 8000
  ```
- [ ] Test HTTP endpoint
  ```bash
  curl -X POST http://localhost:8000/advisory \
       -H "Content-Type: application/json" \
       -d '{"action_type":"read","actor_role":"member","actor_id":"user:test","t3_tensor":{"competence":0.8,"reliability":0.8,"integrity":0.8}}'
  ```
- [ ] Set up systemd service (Linux)
  ```bash
  sudo cp policy-llm.service /etc/systemd/system/
  sudo systemctl enable policy-llm
  sudo systemctl start policy-llm
  ```

**Option B: Direct Integration (Recommended for Web4)**

- [ ] Import in web4 policy module
  ```python
  from llama_cpp import Llama
  from prompts_v4 import build_prompt_v4
  ```
- [ ] Create singleton instance
  ```python
  # See INTEGRATION_GUIDE.md for PolicyAdvisorSingleton
  ```
- [ ] Add to web4.Policy class
  ```python
  # See INTEGRATION_GUIDE.md for web4 integration example
  ```

### 6. Monitoring Setup

- [ ] **Logging configured**
  ```python
  import logging

  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      handlers=[
          logging.FileHandler('/var/log/policy-llm/advisory.log'),
          logging.StreamHandler()
      ]
  )
  ```
- [ ] **Metrics collection** (optional but recommended)
  ```python
  # Track: latency, decision distribution, error rate
  # See INTEGRATION_GUIDE.md monitoring section
  ```
- [ ] **Health check endpoint**
  ```python
  @app.get("/health")
  async def health():
      return {"status": "healthy", "model": "phi-4-mini-7b"}
  ```

---

## Shadow Mode (Week 1-2)

### 7. Shadow Deployment

- [ ] **Deploy alongside existing policy engine**
  - Don't enforce LLM decisions yet
  - Just log advisory opinions
- [ ] **Log all advisories**
  ```python
  logger.info(f"Advisory: {decision} | Situation: {situation_id}")
  ```
- [ ] **Compare with current decisions**
  ```python
  if llm_decision != current_decision:
      logger.warning(f"Mismatch: LLM={llm_decision}, Current={current_decision}")
  ```
- [ ] **Measure agreement rate**
  ```
  Target: >80% agreement in Week 1
  Target: >90% agreement in Week 2
  ```

### 8. Shadow Mode Analysis

- [ ] **Review logs daily**
  - Check for errors, crashes
  - Identify patterns of disagreement
- [ ] **Latency analysis**
  ```python
  # Measure p50, p95, p99
  # Target: p95 < 5s on GPU
  ```
- [ ] **Decision distribution**
  ```
  Allow: ~60-70%
  Deny: ~10-15%
  Require_attestation: ~15-25%
  ```
- [ ] **Edge case collection**
  - Save situations where LLM disagrees with current policy
  - Human review for correctness

---

## Advisory Mode (Week 3)

### 9. Enable Advisory

- [ ] **Surface LLM reasoning to reviewers**
  ```
  UI: Show "LLM recommends: [decision]"
  UI: Show reasoning in expandable section
  ```
- [ ] **Human review workflow**
  - Reviewer sees LLM advisory + reasoning
  - Reviewer makes final decision
  - Log: LLM advisory, human decision, match/mismatch
- [ ] **Feedback collection**
  ```python
  feedback = {
      "situation_id": ...,
      "llm_decision": ...,
      "human_decision": ...,
      "match": llm_decision == human_decision,
      "notes": "..."  # Why they agreed/disagreed
  }
  ```

### 10. Advisory Mode Analysis

- [ ] **Override rate measurement**
  ```
  Override rate = (human decisions != LLM decisions) / total decisions
  Target: <10%
  ```
- [ ] **Categorize overrides**
  - False positives (LLM too strict)
  - False negatives (LLM too permissive)
  - Unclear cases (human uncertain)
- [ ] **Adjust if needed**
  - If override rate >15%: Review prompt or add examples
  - If specific pattern fails: Add training case

---

## Production Rollout (Week 4+)

### 11. Gradual Rollout

**Phase 1: Low-Risk Actions**
- [ ] Enable for **read** actions only
- [ ] Monitor for 3-7 days
- [ ] Override rate < 5%

**Phase 2: Medium-Risk Actions**
- [ ] Enable for **write** and **commit** actions
- [ ] Monitor for 3-7 days
- [ ] Override rate < 10%

**Phase 3: High-Risk Actions**
- [ ] Enable for **deploy** actions (non-production first)
- [ ] Keep **admin actions** in advisory-only mode
- [ ] Monitor for 7-14 days

**Phase 4: Full Deployment**
- [ ] All actions use LLM advisory
- [ ] Admin actions still require human approval (as designed)
- [ ] Continuous monitoring

### 12. Production Monitoring

- [ ] **Uptime tracking**
  ```
  Target: >99% uptime
  Monitor service restarts, crashes
  ```
- [ ] **Latency monitoring**
  ```
  Target: p95 < 5s
  Alert if p95 > 10s
  ```
- [ ] **Error rate**
  ```
  Target: <1% error rate
  Errors = timeouts, crashes, parsing failures
  ```
- [ ] **Decision quality**
  ```
  Track: Allow/deny/attestation distribution
  Alert if distribution shifts >20%
  ```

---

## Post-Deployment

### 13. Continuous Improvement

- [ ] **Weekly review**
  - Check logs for errors
  - Review override cases
  - Identify improvement opportunities
- [ ] **Monthly analysis**
  - Aggregate metrics
  - Compare with baseline (shadow mode)
  - Report findings
- [ ] **Feedback loop**
  - Collect human corrections
  - When 50+ corrections available: Consider fine-tuning
  - Update prompts based on learnings

### 14. Model Updates

- [ ] **New prompt versions**
  - Test in shadow mode first
  - A/B test vs current version
  - Gradual rollout
- [ ] **Model upgrades**
  - Test on validation set
  - Shadow mode for 1-2 weeks
  - Gradual rollout if metrics improve
- [ ] **Rollback plan**
  - Keep previous model version available
  - Quick rollback if issues arise
  - Document rollback procedure

---

## Success Criteria

### Technical Metrics

- ✅ **Uptime**: >99%
- ✅ **Latency**: p95 <5s (GPU) or <15s (CPU)
- ✅ **Error rate**: <1%
- ✅ **Pass rate**: >95% (on test suite)

### Business Metrics

- ✅ **Override rate**: <10% (LLM matches human judgment)
- ✅ **Time to decision**: Reduced by >50% for complex cases
- ✅ **Audit trail quality**: All decisions have reasoning
- ✅ **No critical errors**: Zero wrong allows on high-risk actions

### Operational Metrics

- ✅ **Resource usage**: Within budget (CPU/GPU/memory)
- ✅ **Team satisfaction**: Positive feedback from reviewers
- ✅ **Maintainability**: Clear docs, runbooks, monitoring

---

## Rollback Procedure

### If Issues Arise

1. **Immediate** (Critical issue like wrong allows)
   ```bash
   # Stop LLM service
   sudo systemctl stop policy-llm

   # Revert to previous policy engine
   # (Keep LLM in advisory-only mode)
   ```

2. **Investigate**
   - Review logs for error patterns
   - Identify root cause (model, prompt, integration)
   - Test fix in dev environment

3. **Re-deploy**
   - Fix applied
   - Test in shadow mode again (1 week minimum)
   - Gradual re-rollout

---

## Support & Resources

### Documentation

- **Integration Guide**: `INTEGRATION_GUIDE.md`
- **Session Summaries**: `SESSION_SUMMARY_*.md`
- **Test Suite**: `test_suite_semantic.py`
- **Prompts**: `prompts_v4.py` (recommended)

### Contacts

- **Thor Policy Training Track**: Session logs in `/home/dp/ai-workspace/private-context/autonomous-sessions/`
- **Hardbound Team**: See `hardbound/POLICY_MODEL_SPEC.md`
- **Web4 Team**: See `web4/` repository

### Emergency

- **Model fails to load**: Check path, permissions, disk space
- **High latency**: Verify GPU usage, reduce context window
- **Inconsistent decisions**: Use temperature=0.0 for determinism
- **See**: `INTEGRATION_GUIDE.md` troubleshooting section

---

## Checklist Summary

### Pre-Deployment
- [ ] Environment setup (Python, GPU, disk space)
- [ ] Model downloaded and verified
- [ ] Dependencies installed
- [ ] Configuration set

### Deployment
- [ ] Initial testing (load, inference, latency)
- [ ] Service deployed (sidecar or direct)
- [ ] Monitoring configured
- [ ] Health checks active

### Shadow Mode (Week 1-2)
- [ ] Logging all advisories
- [ ] Comparing with current decisions
- [ ] Analyzing agreement rate (target >80%)
- [ ] Collecting edge cases

### Advisory Mode (Week 3)
- [ ] Surfacing LLM reasoning to humans
- [ ] Collecting feedback
- [ ] Measuring override rate (target <10%)

### Production (Week 4+)
- [ ] Gradual rollout (reads → writes → deploys)
- [ ] Monitoring (uptime, latency, quality)
- [ ] Continuous improvement
- [ ] Success criteria met

---

**Status**: Ready for deployment
**Version**: 1.0
**Last Updated**: 2026-02-04 (Session K complete)
