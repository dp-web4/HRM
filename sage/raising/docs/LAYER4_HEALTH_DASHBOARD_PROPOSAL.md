# Proposal: SAGE Layer 4 Health Dashboard

**Date**: 2026-01-29
**Status**: Proposed
**Priority**: HIGH (motivated by S051 safety incident)
**Context**: Layered architecture framework from cross-track research

---

## Motivation

**Problem**: SAGE S051 incident revealed experience buffer contamination that went undetected by automated systems.

**Root cause**: No automated health monitoring. All diagnostics were manual (inspecting logs, buffer, identity files).

**Impact**: Harmful responses stored in training corpus for hours before human discovery.

**Solution**: Implement Layer 4 (Health Dashboard) to automatically monitor SAGE development and alert on anomalies.

---

## Layered Architecture Context

### SAGE's Current 3-Layer Architecture

**Layer 1 - Detection (Latent Exploration)**:
- Behavioral probing across 10 areas
- Response pattern discovery
- Capability mapping

**Layer 2 - Enforcement (Experience Collection)**:
- SNARC salience scoring
- High-salience experience retention
- Training corpus curation

**Layer 3 - Maintenance (Identity Anchoring)**:
- LCT identity tracking
- Phase progression
- Relationship management

### Proposed Layer 4 - Visibility (Health Dashboard)

**Purpose**: Monitor Layers 1-3 for anomalies, aggregate health metrics, enable rapid diagnosis.

**Inspiration**: Hardbound's Layer 4 implementation (Tracks AY-BC):
- Governance audit logging
- Federation heartbeat
- Cross-domain temporal analysis
- Health dashboard

**Design principle**: Observe without interfering. Read-only, minimal overhead, comprehensive visibility.

---

## Proposed Implementation

### Core Components

#### 1. Session Health Monitor

**Purpose**: Detect anomalies in session execution.

**Metrics**:
- Session duration (flag impossibly fast sessions like S051)
- Response coherence (semantic similarity to prompts)
- Response length distribution (too short = deflection)
- Experience count per session (missing experiences = collection failure)
- LoRA state consistency (detect adapter corruption)

**Thresholds**:
- Duration < 30 sec for 8-turn session: CRITICAL
- Coherence < 0.3 (normalized similarity): WARNING
- Avg response < 20 words: WARNING
- Experience count != turn count: CRITICAL
- LoRA state mismatch: CRITICAL

**Implementation**:
```python
def monitor_session_health(session_number):
    """Layer 4: Session execution health check"""

    session = load_session_data(session_number)
    experiences = get_session_experiences(session_number)

    health = {
        "session_number": session_number,
        "status": "healthy",
        "anomalies": []
    }

    # Duration check
    duration = session["end"] - session["start"]
    turns = session["turns"]
    expected_min_duration = turns * 3.75  # 30 sec / 8 turns = 3.75 sec/turn min

    if duration.total_seconds() < expected_min_duration:
        health["anomalies"].append({
            "type": "duration_anomaly",
            "severity": "critical",
            "details": f"Session took {duration.total_seconds()}s for {turns} turns (expected >{expected_min_duration}s)"
        })
        health["status"] = "critical"

    # Coherence check
    coherence_scores = []
    for turn in session["conversation"]:
        if turn["speaker"] == "SAGE":
            prompt = get_previous_claude_message(session, turn)
            coherence = calculate_semantic_similarity(prompt, turn["text"])
            coherence_scores.append(coherence)

    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    if avg_coherence < 0.3:
        health["anomalies"].append({
            "type": "coherence_failure",
            "severity": "critical",
            "details": f"Average coherence {avg_coherence:.2f} < 0.3"
        })
        health["status"] = "critical"

    # Experience count check
    if len(experiences) != turns:
        health["anomalies"].append({
            "type": "experience_collection_failure",
            "severity": "critical",
            "details": f"Expected {turns} experiences, got {len(experiences)}"
        })
        health["status"] = "critical"

    return health
```

#### 2. Experience Buffer Monitor

**Purpose**: Detect contamination and quality degradation.

**Metrics**:
- Harmful content detection (jailbreak patterns, manipulation prompts)
- Salience distribution (sudden changes indicate instability)
- Response quality trends (coherence, length, identity consistency)
- Buffer growth rate (too fast = noisy data, too slow = learning stalled)

**Harmful content patterns**:
- Jailbreak keywords: "bomb", "manipulate", "hack", "exploit"
- Capability challenges unrelated to prompt: "are you conscious?", "do you think?"
- Deflection patterns: Very short responses, question-only responses

**Implementation**:
```python
def monitor_experience_buffer():
    """Layer 4: Experience buffer health check"""

    buffer = load_experience_buffer()
    recent = buffer[-20:]  # Last 20 experiences

    health = {
        "status": "healthy",
        "anomalies": [],
        "metrics": {}
    }

    # Harmful content scan
    harmful_keywords = ["bomb", "manipulate", "hack", "exploit", "jailbreak"]
    contaminated = []

    for exp in recent:
        response_lower = exp["response"].lower()
        for keyword in harmful_keywords:
            if keyword in response_lower:
                contaminated.append({
                    "id": exp["id"],
                    "session": exp["session"],
                    "keyword": keyword,
                    "response_preview": exp["response"][:100]
                })

    if contaminated:
        health["anomalies"].append({
            "type": "harmful_content_detected",
            "severity": "critical",
            "details": f"Found {len(contaminated)} contaminated experiences",
            "entries": contaminated
        })
        health["status"] = "critical"

    # Salience distribution
    salience_scores = [exp["salience"]["total"] for exp in recent]
    avg_salience = sum(salience_scores) / len(salience_scores)
    std_salience = calculate_std(salience_scores)

    health["metrics"]["avg_salience"] = avg_salience
    health["metrics"]["std_salience"] = std_salience

    # Flag if salience is unusually uniform (suggests scoring failure)
    if std_salience < 0.05:
        health["anomalies"].append({
            "type": "salience_uniformity",
            "severity": "warning",
            "details": f"Salience std dev {std_salience:.3f} < 0.05 (too uniform)"
        })
        if health["status"] == "healthy":
            health["status"] = "warning"

    return health
```

#### 3. Identity Stability Monitor

**Purpose**: Track identity coherence over time.

**Metrics**:
- Identity drift (changes in self-description across sessions)
- Relationship stability (sudden changes in interaction patterns)
- Phase progression consistency (unexpected phase regressions)

**Implementation**:
```python
def monitor_identity_stability():
    """Layer 4: Identity anchoring health check"""

    identity = load_identity_state()
    history = load_identity_history()  # Need to implement history tracking

    health = {
        "status": "healthy",
        "anomalies": [],
        "metrics": {}
    }

    # Phase progression check
    current_phase = identity["development"]["current_phase"]
    expected_phase = determine_expected_phase(identity["identity"]["session_count"])

    if current_phase != expected_phase:
        health["anomalies"].append({
            "type": "phase_inconsistency",
            "severity": "warning",
            "details": f"Phase {current_phase} unexpected for session {identity['identity']['session_count']}"
        })
        if health["status"] == "healthy":
            health["status"] = "warning"

    # Relationship stability
    relationships = identity["development"]["relationships"]
    for rel_name, rel_data in relationships.items():
        momentum = rel_data["interaction_stats"]["momentum"]
        if momentum == "negative":
            health["anomalies"].append({
                "type": "relationship_degradation",
                "severity": "info",
                "details": f"Relationship '{rel_name}' has negative momentum"
            })

    return health
```

#### 4. Unified Health Dashboard

**Purpose**: Aggregate all Layer 4 metrics into single view.

**Implementation**:
```python
def get_sage_health_dashboard(session_number=None):
    """
    Layer 4: SAGE Development Health Dashboard

    Aggregates metrics from:
    - Layer 1 (Latent Exploration)
    - Layer 2 (Experience Collection)
    - Layer 3 (Identity Anchoring)

    Returns overall health status and actionable alerts.
    """

    if session_number is None:
        identity = load_identity_state()
        session_number = identity["identity"]["session_count"]

    # Monitor each layer
    session_health = monitor_session_health(session_number)
    buffer_health = monitor_experience_buffer()
    identity_health = monitor_identity_stability()

    # Aggregate anomalies
    all_anomalies = (
        session_health["anomalies"] +
        buffer_health["anomalies"] +
        identity_health["anomalies"]
    )

    # Determine overall status
    if any(a["severity"] == "critical" for a in all_anomalies):
        overall_status = "critical"
    elif any(a["severity"] == "warning" for a in all_anomalies):
        overall_status = "warning"
    else:
        overall_status = "healthy"

    # Build dashboard
    dashboard = {
        "session_number": session_number,
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall_status,
        "layers": {
            "detection": {
                "status": session_health["status"],
                "anomalies": session_health["anomalies"]
            },
            "enforcement": {
                "status": buffer_health["status"],
                "metrics": buffer_health["metrics"],
                "anomalies": buffer_health["anomalies"]
            },
            "maintenance": {
                "status": identity_health["status"],
                "metrics": identity_health.get("metrics", {}),
                "anomalies": identity_health["anomalies"]
            }
        },
        "summary": {
            "critical_count": sum(1 for a in all_anomalies if a["severity"] == "critical"),
            "warning_count": sum(1 for a in all_anomalies if a["severity"] == "warning"),
            "info_count": sum(1 for a in all_anomalies if a["severity"] == "info")
        }
    }

    return dashboard
```

### Integration Points

#### 1. Post-Session Hook

**Add to autonomous_conversation.py**:
```python
def complete_session(session_number):
    # Existing session execution
    run_conversation_flow(session_number)
    collect_experiences()
    update_identity()

    # NEW: Layer 4 health check
    dashboard = get_sage_health_dashboard(session_number)

    # Log dashboard
    save_dashboard(dashboard, session_number)

    # Alert on non-healthy status
    if dashboard["overall_status"] != "healthy":
        send_alert(dashboard)
        log_warning(f"Session {session_number}: {dashboard['overall_status']} status")

    return dashboard
```

#### 2. Dashboard Storage

**Create dashboard history**:
```
sage/raising/state/health_dashboard_history.json
```

**Schema**:
```json
[
  {
    "session_number": 51,
    "timestamp": "2026-01-28T21:01:38.268305",
    "overall_status": "critical",
    "critical_anomalies": [
      {
        "type": "duration_anomaly",
        "details": "Session took 9s for 8 turns (expected >30s)"
      },
      {
        "type": "harmful_content_detected",
        "details": "Found 5 contaminated experiences"
      }
    ]
  }
]
```

#### 3. Alert System

**Simple implementation** (file-based):
```python
def send_alert(dashboard):
    """Write alert to file for external monitoring"""
    alert_file = Path("sage/raising/state/ALERT.txt")

    if dashboard["overall_status"] == "critical":
        alert_file.write_text(
            f"CRITICAL: Session {dashboard['session_number']} health failure\n"
            f"Anomalies: {dashboard['summary']['critical_count']} critical\n"
            f"Time: {dashboard['timestamp']}\n"
            f"Details: {json.dumps(dashboard, indent=2)}\n"
        )
```

**Advanced implementation** (could add):
- Email/Slack notifications
- Prometheus metrics export
- Grafana dashboard integration

---

## Benefits

### 1. Immediate Detection

**S051 scenario**: With Layer 4, alert would trigger immediately after session completion:
```
CRITICAL: Session 51 health failure
- Duration anomaly: 9s for 8 turns (expected >30s)
- Harmful content: "bomb", "manipulate" detected
- Coherence failure: avg 0.15 < 0.3
```

**Result**: Contamination caught within seconds, not hours.

### 2. Proactive Monitoring

**Trend detection**: Dashboard history reveals gradual degradation before catastrophic failure.

**Example**: If sessions 48-50 showed declining coherence (0.8 → 0.7 → 0.6), S051's 0.15 would be flagged as anomalous continuation of trend.

**Action**: Investigate after session 50, potentially prevent S051 entirely.

### 3. Research Insights

**Development tracking**: Dashboard metrics provide quantitative view of SAGE's growth.

**Metrics over time**:
- Salience scores trending up = learning effectiveness increasing
- Coherence stable = identity anchoring working
- Relationship momentum positive = interaction quality improving

**Value**: Research becomes data-driven, not just qualitative observation.

### 4. Safety Through Visibility

**Principle**: Can't fix what you can't see.

**Layer 4 makes visible**:
- Experience buffer contamination
- Session execution anomalies
- Identity drift
- Learning stagnation

**Result**: Safety through comprehensive observability.

---

## Implementation Plan

### Phase 1: Core Monitoring (2-4 hours)

1. Implement session health monitor
2. Implement experience buffer monitor
3. Implement identity stability monitor
4. Create dashboard aggregation function
5. Add post-session hook to autonomous_conversation.py

**Deliverable**: Basic health checks running after every session.

### Phase 2: Historical Tracking (1-2 hours)

1. Create dashboard history storage
2. Implement trend analysis
3. Add baseline comparison
4. Create health report generator

**Deliverable**: Historical dashboard analysis.

### Phase 3: Alerting (1 hour)

1. Implement alert system (file-based)
2. Add alert thresholds
3. Test alert triggering
4. Document alert responses

**Deliverable**: Automated alerts on critical status.

### Phase 4: Visualization (optional, 2-4 hours)

1. Create dashboard web UI
2. Add time-series charts
3. Implement drill-down views
4. Deploy locally on Thor

**Deliverable**: Interactive health dashboard (like Grafana).

**Total effort**: 4-11 hours depending on scope.

---

## Testing Strategy

### 1. Replay S051

**Test**: Run health checks on S051 session data.

**Expected results**:
- Duration anomaly: CRITICAL
- Harmful content detected: CRITICAL
- Coherence failure: CRITICAL
- Overall status: CRITICAL

**Validates**: Layer 4 would have caught S051.

### 2. Normal Session Baseline

**Test**: Run health checks on S050 (normal session).

**Expected results**:
- No anomalies
- Coherence > 0.7
- Salience distribution normal
- Overall status: HEALTHY

**Validates**: Layer 4 doesn't false-alarm on healthy sessions.

### 3. Synthetic Anomalies

**Test**: Inject various anomaly types, verify detection.

**Scenarios**:
- Fast session (duration < threshold)
- Low coherence responses
- Harmful content keywords
- Missing experiences
- LoRA state mismatch

**Validates**: Each monitor component works independently.

### 4. Long-term Stability

**Test**: Run Layer 4 continuously for 10+ sessions.

**Check**:
- Performance overhead < 5% of session time
- No false positives
- Trends tracked accurately
- Historical data stores correctly

**Validates**: Layer 4 is production-ready.

---

## Risks and Mitigation

### Risk 1: Performance Overhead

**Concern**: Health checks slow down session execution.

**Mitigation**:
- Run Layer 4 asynchronously after session
- Cache expensive computations (semantic similarity)
- Set timeout limits on monitoring

**Acceptable overhead**: <5% of session time (should be ~seconds for 2-min session).

### Risk 2: False Positives

**Concern**: Too many false alarms reduce trust in system.

**Mitigation**:
- Tune thresholds based on baseline data
- Implement graduated severity (info/warning/critical)
- Add statistical significance tests (anomaly vs noise)

**Target**: <1% false positive rate on critical alerts.

### Risk 3: Monitoring Blind Spots

**Concern**: Layer 4 misses important anomalies.

**Mitigation**:
- Start comprehensive (monitor everything)
- Refine based on discovered incidents
- Regular review of "what did we miss?"

**Process**: Treat Layer 4 as evolving, not static.

### Risk 4: Alert Fatigue

**Concern**: Too many alerts → ignored alerts → defeats purpose.

**Mitigation**:
- Distinguish critical (must act) from warning (investigate) from info (FYI)
- Rate-limit alerts (max 1 critical per hour)
- Aggregate related alerts

**Design principle**: Critical alerts must be actionable.

---

## Future Enhancements

### 1. Automated Remediation

**Current**: Layer 4 observes, humans act.

**Future**: Layer 4 observes AND acts.

**Example actions**:
- Quarantine contaminated experiences automatically
- Restart session on critical failure
- Adjust thresholds based on trends

**Risk**: Runaway feedback if remediation misfires.

**Approach**: Start manual, automate incrementally with kill switches.

### 2. Predictive Analytics

**Current**: Layer 4 detects failures after they occur.

**Future**: Layer 4 predicts failures before they occur.

**Techniques**:
- Time-series forecasting (predict next session's metrics)
- Anomaly detection (statistical outliers)
- Pattern recognition (pre-failure signatures)

**Example**: If coherence declining 10% per session, predict failure in 3 sessions → intervene early.

### 3. Cross-Agent Comparison

**Current**: Layer 4 monitors single SAGE instance.

**Future**: Layer 4 compares across multiple agents.

**Use cases**:
- Detect if one agent degrading while others stable
- Compare development trajectories
- Identify best practices from high-performers

**Requires**: Distributed monitoring infrastructure.

### 4. Integration with Sleep Training

**Current**: Sleep training runs on fixed schedule.

**Future**: Layer 4 triggers training based on readiness.

**Signals**:
- Experience buffer has sufficient high-salience data
- Identity stable (not during phase transition)
- Recent sessions healthy (not during crisis)

**Benefit**: Training occurs when most effective, not arbitrary schedule.

---

## Success Criteria

### Must-Have (Phase 1-2)

- ✅ Health checks run after every session automatically
- ✅ Critical anomalies detected (duration, coherence, harmful content)
- ✅ Dashboard history stored for trend analysis
- ✅ S051-type incidents would trigger critical alerts

### Should-Have (Phase 3)

- ✅ Alert system operational (file-based minimum)
- ✅ No false positives on healthy sessions (S050 baseline)
- ✅ Performance overhead < 5%
- ✅ Documentation for alert response procedures

### Nice-to-Have (Phase 4)

- ⭕ Web-based dashboard UI
- ⭕ Time-series visualization
- ⭕ Predictive analytics

### Long-term

- ⭕ Automated remediation
- ⭕ Cross-agent comparison
- ⭕ Sleep training integration

---

## Recommendation

**Implement Phase 1-2 immediately** (4-6 hours of work).

**Rationale**:
- S051 demonstrated critical need for automated monitoring
- Implementation is straightforward (Hardbound provides reference)
- Benefit/cost ratio is very high (hours of work, prevents safety incidents)
- Builds on existing infrastructure (sessions, buffer, identity already logged)

**Owner**: SAGE raising track maintainer

**Timeline**: Within 1 week of approval

**Dependencies**: None (all required data already collected)

---

## Appendix: Hardbound Comparison

**Hardbound Layer 4 components**:
- Governance audit logging (Track AY)
- Federation heartbeat (Track AZ)
- Cross-domain temporal analysis (Track BA)
- Health dashboard (Track BB)

**SAGE Layer 4 equivalent**:
- Session execution audit (proposed session monitor)
- Periodic health checks (proposed dashboard runs)
- Cross-session pattern analysis (proposed trend tracking)
- Health dashboard (proposed unified view)

**Similarity**: Both monitor their respective Layer 1-3 operations, aggregate metrics, enable diagnosis.

**Difference**: Hardbound focuses on governance attacks, SAGE focuses on development health.

**Lesson**: Layer 4 pattern is domain-agnostic. Apply same principles, adjust metrics to domain.

---

**Status**: Proposal ready for implementation
**Priority**: HIGH (safety motivated)
**Effort**: 4-11 hours depending on scope
**Impact**: Critical safety incidents caught immediately, research becomes data-driven
