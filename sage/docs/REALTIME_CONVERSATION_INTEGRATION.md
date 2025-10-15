# Real-Time Conversation Integration with SAGE Unified

**Date**: October 14, 2025
**Status**: Design Phase
**Goal**: Integrate real-time conversation system into SAGE's continuous consciousness loop

## Architecture Analysis

### Current System State

**Working Components** (Standalone):
1. `StreamingAudioSensor` - VAD-based audio capture with faster-whisper transcription
2. `CognitiveMailbox` - Memory-based IPC (queue fallback implemented)
3. `PatternResponseEngine` - Fast pattern matching for cognitive responses
4. `Piper TTS` - Neural text-to-speech synthesis
5. Integration test demonstrating end-to-end conversation (2.2s latency)

**SAGE Unified Loop** (`sage_unified.py`):
```python
while running:
    1. SENSE: observations = sensor_hub.poll()
    2. EVALUATE: salience_scores = hierarchical_snarc.score_all(observations)
    3. ALLOCATE: atp_allocations = _allocate_atp(salience_scores, metabolic_config)
    4. REFINE: irp_results = _run_irp(observations, atp_allocations)
    5. LEARN: _update_trust(irp_results)
    6. REMEMBER: _store_memory(observations, irp_results)
    7. METABOLIZE: metabolic_controller.update(...)
```

### Integration Points Identified

#### 1. **SensorHub Integration** - Audio as First-Class Sensor
- `StreamingAudioSensor` inherits from `BaseSensor`
- Already implements `poll()` → returns `SensorReading`
- **Action**: Register with `SAGEUnified.register_sensor()`
- **Integration**: `sensor_hub.poll()` will include audio transcriptions

#### 2. **SNARC Salience for Audio** - Speech Event Scoring
- Need `SensorSNARC` for audio modality
- Score dimensions for speech:
  - **Surprise**: Unexpected speech vs silence
  - **Novelty**: New speaker or topic
  - **Arousal**: Volume, urgency, emotion
  - **Reward**: Successful transcription confidence
  - **Conflict**: Ambiguous/low-confidence transcription
- **Action**: Create `AudioSNARC` class or use generic `SensorSNARC`

####3. **IRP Plugin for Conversation** - Pattern Matching as IRP
- `PatternResponseEngine` can be wrapped as IRP plugin
- Iterative refinement: Try patterns in priority order
- Energy function: `1.0 - match_confidence`
- Halt condition: Pattern found OR all patterns exhausted
- **Action**: Create `conversation_irp.py` implementing IRP interface

#### 4. **Effector Integration** - TTS as SAGE Effector
- Piper TTS needs `EffectorHub` integration
- Audio effector executes speech synthesis
- Non-blocking: spawn subprocess for synthesis
- **Action**: Create `tts_effector.py` for Piper

#### 5. **CognitiveMailbox Integration** - Replace Queue with GPU Mailbox
- Current: Queue-based fallback works
- Future: GPU mailbox PBM for cross-component IPC
- Transcriptions → Cognitive layer (Group 0)
- Responses → TTS layer (Group 1)
- **Action**: Already working, GPU mailbox is optimization

## Integration Design

### Phase 1: Sensor Registration (Minimal Integration)

**Goal**: Get audio transcriptions into SAGE loop

**Implementation**:
```python
# In sage_unified.py or startup script
audio_sensor = StreamingAudioSensor({
    'sensor_id': 'conversation_audio',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'vad_aggressiveness': 2,
    'whisper_model': 'tiny'
})

sage.register_sensor(audio_sensor)
```

**Result**:
- Audio transcriptions appear in `observations` dict
- SNARC scores audio events
- ATP allocated based on speech salience
- No response yet (just logging)

**Test**: Verify transcriptions logged in SAGE cycle output

---

### Phase 2: Conversation IRP Plugin (Cognitive Response)

**Goal**: Generate responses using pattern matching within IRP framework

**New File**: `sage/irp/plugins/conversation_impl.py`

**Implementation**:
```python
class ConversationIRP:
    """IRP plugin for conversational responses"""

    def init_state(self, x0, task_ctx):
        return IRPState(
            x=x0,  # Transcription text
            metadata={'patterns_tried': [], 'best_match': None}
        )

    def step(self, state):
        """Try next pattern"""
        # Get transcription text from state
        text = state.metadata.get('transcription', '')
        patterns_tried = state.metadata['patterns_tried']

        # Try next untried pattern
        pattern, response = pattern_engine.get_next_pattern(text, patterns_tried)

        if pattern:
            patterns_tried.append(pattern)
            confidence = pattern_engine.match_confidence(pattern, text)

            if confidence > state.metadata.get('best_confidence', 0):
                state.metadata['best_match'] = {
                    'pattern': pattern,
                    'response': response,
                    'confidence': confidence
                }
                state.metadata['best_confidence'] = confidence

        return state

    def energy(self, state):
        """Energy = 1.0 - best_confidence"""
        return 1.0 - state.metadata.get('best_confidence', 0.0)

    def halt(self, history):
        """Halt if good match OR all patterns tried"""
        current = history[-1]
        best_confidence = current.metadata.get('best_confidence', 0.0)

        # Halt if good match (confidence > 0.7)
        if best_confidence > 0.7:
            return True

        # Halt if all patterns tried
        if len(current.metadata['patterns_tried']) >= pattern_engine.num_patterns:
            return True

        # Continue if more iterations available
        return len(history) >= 10  # Max 10 pattern attempts
```

**Integration**:
```python
# Register IRP plugin
conversation_plugin = ConversationIRP()
sage.register_irp_plugin('conversation_audio', conversation_plugin)
```

**Result**:
- Pattern matching runs as iterative refinement
- Energy decreases as better matches found
- Trust scores based on convergence quality
- Response text in `irp_results['conversation_audio']`

---

### Phase 3: TTS Effector (Voice Output)

**Goal**: Synthesize and play responses through Piper TTS

**New File**: `sage/interfaces/tts_effector.py`

**Implementation**:
```python
class TTSEffector:
    """Piper TTS effector for speech synthesis"""

    def __init__(self, config):
        self.piper_path = config.get('piper_path', '/home/sprout/ai-workspace/piper/piper/piper')
        self.model_path = config.get('model_path', '/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx')
        self.bt_sink = config.get('bt_sink', 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit')
        self.processes = []  # Track background processes

    def execute(self, text: str):
        """Synthesize and play text (non-blocking)"""
        if not text:
            return

        # Pipe text through Piper to paplay
        piper_proc = subprocess.Popen(
            [self.piper_path, "--model", self.model_path, "--output_raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        play_proc = subprocess.Popen(
            ["paplay",
             "--device", self.bt_sink,
             "--rate", "22050",
             "--format", "s16le",
             "--channels", "1",
             "--raw"],
            stdin=piper_proc.stdout,
            stderr=subprocess.DEVNULL
        )

        # Send text (non-blocking)
        threading.Thread(
            target=self._send_and_cleanup,
            args=(piper_proc, play_proc, text),
            daemon=True
        ).start()

    def _send_and_cleanup(self, piper_proc, play_proc, text):
        """Send text and wait for completion"""
        try:
            piper_proc.stdin.write(text.encode('utf-8'))
            piper_proc.stdin.close()
            play_proc.wait(timeout=10)
        except Exception as e:
            print(f"TTS error: {e}")
```

**Integration**:
```python
# In SAGE unified loop, after IRP results
def cycle(self):
    # ... existing code ...

    # After IRP refinement
    irp_results = self._run_irp(observations, atp_allocations)

    # Extract responses and send to TTS
    for sensor_id, result in irp_results.items():
        if sensor_id == 'conversation_audio':
            response_text = result.get('response', None)
            if response_text:
                self.tts_effector.execute(response_text)

    # ... rest of cycle ...
```

---

### Phase 4: ATP-Based Resource Allocation (Full Integration)

**Goal**: Conversation competes for ATP with other modalities

**Mechanism**:
- Audio salience computed by `SensorSNARC`
- High salience (speech detected) → more ATP
- Low salience (silence) → minimal ATP
- Pattern IRP uses ATP for iteration budget
- TTS uses ATP for synthesis priority

**ATP Flow**:
```
Speech Event (high salience: 0.8)
  ↓
ATP Allocation: 0.8 × trust_score × metabolic_config
  ↓
IRP Iterations: min(ATP / 2, 10) attempts
  ↓
TTS Execution: Prioritized if ATP remaining
```

**Metabolic States**:
- **WAKE**: Full conversation mode (70% attention breadth)
- **FOCUS**: Conversation paused (30% attention breadth, vision priority)
- **REST**: No conversation (10% attention breadth)
- **DREAM**: Processing conversation memories
- **CRISIS**: High priority for alerts only

---

### Phase 5: Memory Integration

**Goal**: Conversation history in SAGE memory systems

**Memory Types**:
1. **Circular Buffer** - Recent 10 exchanges (X-from-last)
2. **SNARC Selective** - High-salience conversations
3. **IRP Success Library** - Successful pattern matches
4. **Verbatim Storage** - Full transcript (SQLite)

**Implementation**:
```python
def _store_conversation_memory(self, transcription, response, salience):
    """Store conversation in all memory systems"""

    # Circular buffer (always)
    self.memory.circular_buffer.append({
        'transcription': transcription,
        'response': response,
        'timestamp': time.time()
    })

    # SNARC selective (if salient)
    if salience > 0.7:
        self.memory.snarc_memory.store({
            'event': 'conversation',
            'content': transcription,
            'salience': salience
        })

    # IRP success library (if pattern matched)
    if response:
        self.memory.irp_bridge.store_episode(
            inputs={'text': transcription},
            outputs={'response': response},
            success=True
        )
```

---

## Implementation Checklist

### Core Integration (Minimum Viable)
- [ ] Register StreamingAudioSensor with SensorHub
- [ ] Create AudioSNARC or configure generic SensorSNARC
- [ ] Implement ConversationIRP plugin
- [ ] Create TTSEffector class
- [ ] Wire IRP results → TTS effector
- [ ] Test conversation within SAGE loop

### Enhanced Features
- [ ] ATP-based conversation priority
- [ ] Metabolic state awareness (pause in FOCUS/REST)
- [ ] Conversation memory integration
- [ ] Multi-turn context tracking
- [ ] LLM fallback for complex queries (future)

### Performance Optimization
- [ ] GPU mailbox for CognitiveMailbox (replace queue)
- [ ] Piper TTS latency profiling and optimization
- [ ] Parallel transcription and TTS
- [ ] Pattern caching for repeat phrases

---

## Expected Integration Benefits

### 1. **Attention Orchestration**
- Speech automatically gets ATP when detected
- Conversation pauses during high-priority vision tasks
- Metabolic states control conversation availability

### 2. **Trust-Based Learning**
- Pattern matches that converge quickly → higher trust
- Unreliable patterns → lower ATP allocation
- System learns which patterns work best

### 3. **Memory Consolidation**
- Important conversations stored automatically
- Pattern library improves over time
- Dream state can replay and analyze conversations

### 4. **Resource Management**
- Whisper model loads only when needed
- TTS resources freed when not conversing
- Dynamic memory management across all modalities

### 5. **Unified Metrics**
- Conversation latency visible in SAGE cycle times
- ATP usage tracked alongside other modalities
- Trust scores quantify conversation quality

---

## Integration Sequence (Recommended)

### Session 1: Basic Integration
1. Register audio sensor
2. Verify transcriptions in SAGE loop
3. Log transcriptions (no response yet)
4. Test with 10-20 SAGE cycles

### Session 2: Response Generation
1. Implement ConversationIRP
2. Register IRP plugin
3. Test pattern matching within loop
4. Verify energy convergence

### Session 3: TTS Integration
1. Implement TTSEffector
2. Wire IRP results → effector
3. Test end-to-end conversation in loop
4. Measure total latency

### Session 4: Optimization & Memory
1. Profile bottlenecks
2. Add conversation memory
3. Tune ATP allocations
4. Test metabolic state transitions

---

## Success Criteria

### Functional
- [ ] Audio transcriptions appear in SAGE loop
- [ ] Pattern matching generates responses
- [ ] TTS plays responses through Bluetooth
- [ ] End-to-end conversation works within loop

### Performance
- [ ] Conversation doesn't block other sensors
- [ ] ATP allocation prevents resource starvation
- [ ] Metabolic states correctly modulate conversation
- [ ] Trust scores reflect pattern quality

### Quality
- [ ] Pattern matches are relevant
- [ ] No crashes or hangs during conversation
- [ ] Clean startup and shutdown
- [ ] Logging provides clear visibility

---

## Risks and Mitigations

### Risk: Blocking SAGE Loop
**Issue**: Long transcription times could block other sensors
**Mitigation**: Transcription already happens in background thread, poll() is non-blocking

### Risk: Resource Contention
**Issue**: Whisper + TTS might consume too much memory
**Mitigation**: Dynamic loading (already planned), metabolic states can pause conversation

### Risk: Poor Pattern Matching
**Issue**: Most utterances might not match patterns
**Mitigation**: Expected behavior - log misses, add LLM fallback later

### Risk: TTS Latency
**Issue**: 2.2s TTS time might feel slow
**Mitigation**: Already identified in results, optimize separately

---

## Next Steps

**Immediate** (This Session):
1. Create `conversation_impl.py` IRP plugin
2. Create `tts_effector.py`
3. Register audio sensor with SAGE
4. Test basic integration

**Short-term** (Next Session):
1. Tune ATP allocations
2. Add conversation memory
3. Profile and optimize
4. Document learnings

**Long-term** (Future):
1. LLM fallback for complex queries
2. Multi-turn context
3. GPU mailbox integration
4. Voice emotion detection

---

**Integration Philosophy**: Start simple, verify each layer, build incrementally. The standalone conversation system works - now we're teaching SAGE to orchestrate it alongside other modalities.
