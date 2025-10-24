# Overnight Work Notes - Hybrid Learning Dashboard

## Issues Identified

### 1. Dashboard Not Updating (Static Display)
**Problem**: Dashboard renders once but doesn't update in real-time
**Root Cause**: Terminal rendering is synchronous and only happens during events
**Possible Solutions to Implement**:
- [ ] Use threading to render dashboard in separate thread
- [ ] Use curses for better terminal control
- [ ] Add periodic refresh even when no events
- [ ] Consider web-based dashboard (Flask/FastAPI endpoint)

### 2. Only Fast Path Engaging
**Problem**: LLM slow path not being triggered even for novel questions
**Root Causes**:
1. Pattern engine has 13 default patterns that match too broadly
2. Pattern matching may be too greedy
**Solutions to Implement**:
- [ ] Review default patterns - make them more specific
- [ ] Add pattern confidence threshold (only use if >0.8 confidence)
- [ ] Test with questions that should NOT match patterns
- [ ] Add deliberate pattern miss testing

### 3. Conversation History Format Bug
**Fixed**: Changed MockLLM parameter from `history=` to `conversation_history=`
**Fixed**: Added conversation history update to fast path (was only updating on slow path)

## Planned Overnight Work

### Priority 1: Fix Dashboard Updates
```python
# Option A: Threading approach
import threading

class LiveDashboard:
    def __init__(self):
        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)

    def _render_loop(self):
        while self.running:
            self.render()
            time.sleep(0.1)  # Update 10 times per second

    def start(self):
        self.render_thread.start()
```

### Priority 2: Pattern Confidence Gating
```python
# In pattern_responses.py
def generate_response(self, text: str, min_confidence: float = 0.8) -> tuple:
    """Return (response, confidence) or (None, 0.0)"""
    for pattern, response_variants in self.compiled_patterns:
        match = pattern.search(text)
        if match:
            # Calculate match quality
            match_quality = len(match.group()) / len(text)
            if match_quality >= min_confidence:
                return (random.choice(response_variants), match_quality)
    return (None, 0.0)
```

### Priority 3: Test Pattern Learning
Create test questions that should force LLM:
- "Explain quantum entanglement"
- "What's the meaning of life?"
- "How do neural networks work?"
- "Tell me about black holes"

These should NOT match any default patterns and force slow path.

### Priority 4: Better Logging
Add detailed logging to track:
- Which patterns are matching
- Why fast vs slow path was chosen
- Pattern confidence scores
- Learning events (when new patterns extracted)

### Priority 5: Dashboard Enhancements
If web-based dashboard:
```python
from flask import Flask, render_template_string
import json

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def status():
    return json.dumps({
        'state': dashboard.current_state,
        'stats': hybrid_system.get_stats(),
        'last_conversation': {
            'user': dashboard.last_user_input,
            'sage': dashboard.last_response
        }
    })

# Run in background thread
threading.Thread(target=lambda: app.run(port=5000), daemon=True).start()
```

## Tomorrow's Testing Plan

1. **Dashboard Update Test**: Verify real-time updates work
2. **Pattern Gating Test**: Confirm only high-confidence fast path
3. **Learning Test**: Ask novel questions, verify pattern learning
4. **Integration Test**: Full conversation with learning progression
5. **Visualization Test**: Watch fast path ratio improve over time

## Expected Outcomes

- Dashboard updates smoothly during conversation
- LLM engages for novel questions
- Pattern learning visible in real-time
- Fast path ratio improves from ~0% → 50%+ over 20 questions
- Clean visualization of the learning process

## Files to Work On

1. `/home/sprout/ai-workspace/HRM/sage/tests/hybrid_conversation_realtime.py` - Main integration
2. `/home/sprout/ai-workspace/HRM/sage/cognitive/pattern_responses.py` - Pattern matching
3. `/home/sprout/ai-workspace/HRM/sage/cognitive/pattern_learner.py` - Learning logic
4. Create: `/home/sprout/ai-workspace/HRM/sage/tests/hybrid_web_dashboard.py` - Web version

## Success Criteria

✅ Dashboard updates in real-time during conversation
✅ Both fast and slow paths engage appropriately
✅ Pattern learning observable (new patterns added)
✅ Fast path ratio increases over conversation
✅ Clean, informative visualization

## Notes

- Current system works but needs polish
- Learning logic is sound, just needs to be triggered
- Pattern engine may be too permissive
- Dashboard concept is good, execution needs improvement
- Real-time audio working perfectly
- LLM integration working (Qwen 0.5B on CPU)

The foundation is solid - just needs refinement!
