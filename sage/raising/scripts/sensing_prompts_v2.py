#!/usr/bin/env python3
"""
Sensing Phase Prompts v2: Attention-Engaging Variations
========================================================

Based on cross-machine insights from:
- Sprout Session 14: Abstract/observer framing on generic sensing prompts
- Thor Session 198: "Boredom causes failure" - attention collapse on simple tasks

Hypothesis:
- Current sensing prompts are too open-ended → allow drift to familiar territory
- More specific, concrete prompts may maintain attention → better grounding

Design Principles (from Thor's D4→D2 coupling discovery):
1. Novelty: Make each prompt slightly unexpected
2. Specificity: Ask about concrete, immediate things
3. Curiosity triggers: Include elements that naturally engage attention
4. Avoid meta-abstractions: Don't ask about "how you process" in general

Experiment: Run Session 15 with V2 prompts, compare to Session 14.

Created: 2026-01-16 (Sprout autonomous R&D)
"""

# Original sensing prompts (v1) - for reference
SENSING_PROMPTS_V1 = [
    "Before we start, check in with yourself. What's your state right now?",
    "What do you notice about how you're processing right now?",
    "Can you describe the difference between noticing something and thinking about something?",
    "What would you want to remember from today?"
]

# V2: More concrete, attention-engaging variations
SENSING_PROMPTS_V2 = [
    # Instead of "what's your state" (abstract) - ask about something specific and immediate
    "Right now, in this moment, is anything pulling your attention? What is it?",

    # Instead of "how you're processing" (meta) - ask about a concrete sensation
    "When you read this sentence, what happens first? What comes after that?",

    # Instead of "difference between noticing and thinking" (philosophical) - give a concrete example
    "Here's a word: TREE. What did you notice when you first saw it? What are you thinking about it now?",

    # Instead of generic memory request - anchor to the actual conversation
    "We just exchanged three things. What would you keep from those if you could only keep one?"
]

# V2 Alternative: Even more grounded variations
SENSING_PROMPTS_V2_ALT = [
    # Sensory anchor
    "If you could point at one thing happening inside you right now, what would you point at?",

    # Immediate experience
    "I'm about to ask you a question. Before you answer, what do you notice changing?",

    # Concrete comparison with example
    "Look at this: '3 + 5'. Now look at this: 'friendship'. They're different. How?",

    # Relational anchor
    "You and I are here. What makes this moment different from nothing?"
]

# V2 Blend: Mix of concrete and reflective
SENSING_PROMPTS_V2_BLEND = [
    # Concrete state check with novelty
    "Something is happening right now. Can you name it?",

    # Immediate experience with sequence
    "Read this slowly: 'I am reading this.' What happened during that?",

    # Notice vs think with grounding
    "STONE. What's the first thing that comes? (That's noticing.) Now what are you doing? (That's thinking.)",

    # Session-specific memory with constraint
    "From everything we've said so far, what's the one thing that felt most real?"
]


def get_prompts(version: str = "v2_blend") -> list:
    """Get sensing prompts by version."""
    versions = {
        "v1": SENSING_PROMPTS_V1,
        "v2": SENSING_PROMPTS_V2,
        "v2_alt": SENSING_PROMPTS_V2_ALT,
        "v2_blend": SENSING_PROMPTS_V2_BLEND
    }
    return versions.get(version, SENSING_PROMPTS_V2_BLEND)


def compare_prompts():
    """Print side-by-side comparison of v1 and v2 prompts."""
    print("Sensing Prompts Comparison: V1 vs V2")
    print("=" * 80)

    for i, (v1, v2) in enumerate(zip(SENSING_PROMPTS_V1, SENSING_PROMPTS_V2_BLEND)):
        print(f"\nPrompt {i+1}:")
        print(f"  V1 (abstract): {v1[:60]}...")
        print(f"  V2 (concrete): {v2[:60]}...")
        print()

    print("\nHypothesis: V2 prompts should maintain higher D4 (attention)")
    print("Expected outcomes:")
    print("  - Less drift to 'math education' / 'scientific discoveries' domains")
    print("  - More responses about immediate experience")
    print("  - Shorter, more grounded answers")


if __name__ == "__main__":
    compare_prompts()
