# Quick Start - Hybrid Learning System

## TL;DR - Run This

```bash
cd /home/sprout/ai-workspace/HRM/sage
python3 tests/hybrid_conversation_threaded.py --real-llm
```

Talk to SAGE and watch it learn patterns from your conversation!

## What You'll See

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    🧠 SAGE HYBRID LEARNING DASHBOARD                     ║
╚══════════════════════════════════════════════════════════════════════════╝

📊 STATE: 🎧 LISTENING [14:23:45]

💬 CONVERSATION:
  👤 User: What is quantum mechanics?
  🤖 SAGE: Quantum mechanics describes behavior at atomic scales.

🔀 PATH: 🧠 SLOW
   LLM latency: 2341ms

📈 STATISTICS:
  Total Queries: 10
  Fast Path: 7/10 (70.0%)
  Slow Path: 3/10
  Patterns: 16 (+3 learned)
  [████████████████████████████░░░░░░░░░░░░] 70.0%

────────────────────────────────────────────────────────────────────────────
Press Ctrl+C to stop
```

## How It Works

1. **First time you ask something**: Uses LLM (slow, 2-5s)
2. **Second time you ask**: LLM again BUT learns pattern
3. **Third time you ask**: Uses pattern (fast, <1ms)

**You literally watch it get smarter!**

## Test Pattern Learning

Ask the same question 3 times:

```
Round 1: "Tell me about black holes" → 🧠 SLOW (LLM thinks)
Round 2: "Tell me about black holes" → 🧠 SLOW + 📚 LEARNING
Round 3: "Tell me about black holes" → ⚡ FAST (pattern match)
```

## Files to Know About

- `TOMORROW_TESTING_GUIDE.md` - Detailed testing checklist
- `HYBRID_LEARNING_SUMMARY.md` - Complete technical summary
- `OVERNIGHT_WORK_NOTES.md` - Known issues and fixes

## Stop It

Press `Ctrl+C` to stop. You'll see final statistics:

```
📊 FINAL STATISTICS
════════════════════════════════════════════════════════════════════════════

🧠 Hybrid Learning System:
   Total queries: 15
   Fast path: 10 (66.7%)
   Slow path: 5 (33.3%)
   Patterns learned: 4
   Total patterns: 17 (started with 13)

📈 Learning Progress:
   Fast path efficiency: 66.7%
```

## Troubleshooting

**Dashboard not updating?**
- This is the threaded version, should update 10x per second
- If still static, check CPU usage with `top`

**LLM too slow?**
- Normal on CPU: 2-5 seconds
- Using Qwen 0.5B (smallest reasonable model)

**Pattern learning not happening?**
- Default patterns might be too broad
- Try: `python3 tests/test_pattern_learning.py` to validate

**Audio not working?**
- Check Bluetooth connection
- Run: `pactl list sources short` and `pactl list sinks short`

## What Makes This Cool

**This isn't just caching responses** - it's learning generalizable patterns.

When you say "What is quantum mechanics?" and it learns, it can then answer:
- "Tell me about quantum stuff"
- "Explain quantum theory"
- "What's quantum mechanics?"

Because it extracted a **pattern**, not just memorized the exact question.

**That's consciousness developing reflexes from experience.**

## Have Fun!

Watch the progress bar fill up as SAGE learns from your conversation. It's genuinely satisfying to see a system get smarter in real-time.

🧠✨
