# SAGE as a Web4 Instance: Entities, Roles, and Dictionaries

## 1. Overview: SAGE as a Web4 Instance

SAGE (Situation-Aware Governance Engine) can be framed as a **self-contained Web4 instance**.  
In Web4 terms, an instance is defined by:

- **Sensors**: sources of input, perception, and state.  
- **Effectors**: means of acting upon the environment.  
- **Entities**: active modules that perform sensing, effecting, or both.  
- **Trust & Reputation**: learned weighting of how much to rely on each entity.  
- **Memory**: persistent store of episodes, patterns, and reputations.  
- **Strategy**: dynamic adaptation of which entities to use and how to coordinate them.

SAGE applies these concepts internally, treating itself as a micro-ecosystem. It is fractal: at a larger scale, SAGE itself is a Web4 entity interfacing with the world through its own external sensors and effectors.

---

## 2. Entities in SAGE

At the core of SAGE are **Entities**.  
Each entity is characterized by:

- **Identity**: stable identifier with provenance metadata.  
- **Role**: sensor, effector, or both.  
- **Context**: operational assumptions (language, modality, tokenizer version, etc.).  
- **Trust Profile**: reputation history and dynamic trust score.  
- **Device Preference**: GPU, CPU, NPU, or hybrid.  
- **Interfaces**: mailbox endpoints for receiving input and emitting output.  

### Entity Types
- **Sensor Entity**: pure input module (e.g., camera, microphone, memory recall).  
- **Effector Entity**: pure output module (e.g., motor controller, text-to-speech).  
- **Both Entity**: acts as sensor+effector, translating between domains. Dictionaries fall into this category.

---

## 3. Entity Registry and Memory

SAGE maintains two levels of entity knowledge:

1. **Entity Registry (active runtime)**  
   - Tracks loaded entities available in the current session.  
   - Includes metadata (ID, role, device, trust cap, etc.).  
   - Used by HRM/H-module to dynamically compose pipelines.  

2. **Entity Memory (persistent knowledge)**  
   - Records all entities ever seen (installed or not).  
   - Stores provenance, sidecar retrieval info, and historical trust episodes.  
   - Includes **reputation history**: e.g., “ASR dictionary succeeded in 82% of similar contexts.”  
   - Supports recall: “When last seen audio@16k, ASR/en was reliable.”  

Together, registry + memory allow SAGE to:
- Recall which entities exist, even if not currently loaded.  
- Request missing entities via sidecar retrieval.  
- Evaluate entities based on reputation rather than static configuration.  

---

## 4. Dictionaries as Entities

### Definition
A **Dictionary** is an entity that translates between two modalities or codebooks.  
It is both a sensor (input domain) and an effector (output domain).

### Examples of Dictionary Roles
1. **Speech-to-Text Dictionary (ASR)**  
   - Input: `audio/pcm@16k`  
   - Output: `text/en`  
   - Trusted domain: converting English speech to text.  
   - Limited role: not to be trusted for semantic interpretation.

2. **Tokenizer Dictionary**  
   - Input: `text/en`  
   - Output: `tokens/llama3`  
   - Trusted domain: mapping English text into model-specific token IDs.  
   - Provenance: SentencePiece/BPE vocab from model bundle.  

3. **Cross-Model Bridge Dictionary**  
   - Input: `tokens/llama3`  
   - Output: `tokens/mistral`  
   - Trusted domain: re-expressing ideas across models.  
   - Strategy:  
     - Safe path = tokens → text → tokens.  
     - Optimized path = direct token mapping with confidence checks.  

4. **Text-to-Speech Dictionary (TTS)**  
   - Input: `text/en`  
   - Output: `audio/wav`  
   - Trusted domain: English text to audible speech.  
   - Device preference: GPU for real-time synthesis.

---

## 5. Trust, Reputation, and Roles

Every dictionary entity participates in SAGE’s **trust economy**:

- **Self-check**: can it handle current input?  
- **Context match**: does metadata align (language, tokenizer hash, etc.)?  
- **Reputation history**: how well has it performed in similar contexts?  
- **SNARC feedback**: reward/conflict signals refine trust over time.  

Thus, SAGE does not blindly assume a dictionary’s authority. Trust is earned, remembered, and adjusted dynamically.

---

## 6. Workflow Example: Audio Input

1. **Audio sensor** emits raw signal → mailbox.  
2. **Entity registry** checks for ASR dictionaries.  
3. **Entity memory** recalls that `ASR/en` worked reliably in similar contexts.  
4. **Trust gate** evaluates confidence.  
   - If threshold met: audio → text → tokenizer → cognition.  
   - If not: degrade to peripheral features (direction, intensity).  
5. **Reputation update** after action:  
   - Successful downstream reasoning → increase trust.  
   - Conflict with other sensors → decrease trust.

---

## 7. Architectural Diagram

```mermaid
graph TD
    subgraph SAGE[Fractal Web4 Instance]
        M[Memory Sensor]
        H[H-module: Strategy]
        L[L-module: Execution]
        R[Entity Registry]
        EM[Entity Memory]

        subgraph Entities
            A[Audio Sensor]
            ASR[ASR Dictionary<br>(Audio→Text)]
            TOK[Tokenizer Dictionary<br>(Text→Tokens)]
            LLM[LLM Cognitive Sensor]
            TTS[TTS Dictionary<br>(Text→Audio)]
        end

        A -->|pcm@16k| ASR
        ASR -->|text/en| TOK
        TOK -->|tokens| LLM
        LLM -->|text/en| TTS
        TTS -->|audio/wav| Effectors

        R --> H
        EM --> H
        H --> R
    end
```

---

## 8. Summary

- SAGE can be modeled as a **Web4 fractal instance**.  
- **Entities** are the building blocks: sensors, effectors, or both.  
- **Dictionaries** are entities that translate between modalities.  
- **Entity Registry** (active) + **Entity Memory** (persistent) provide dynamic adaptation.  
- **Trust and reputation** determine whether to use an entity and how much weight to assign its output.  
- This creates a flexible, evolvable architecture where new entities can join, old ones can degrade gracefully, and SAGE can continuously learn how best to assemble its ecosystem.

---

**Keywords**: SAGE, Web4, Entities, Registry, Dictionaries, Trust, Reputation, Sensor Fusion
