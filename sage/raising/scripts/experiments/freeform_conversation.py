#!/usr/bin/env python3
"""
Freeform Conversation with SAGE

A genuine multi-turn conversation that breaks out of the scripted
6-prompt session runner. Each turn builds on SAGE's actual response,
following threads rather than following a script.

Turns 2 and 5 are dynamically constructed based on what SAGE says.
Conversation history accumulates so SAGE has full context.
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime

# Resolve paths
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # experiments -> scripts -> raising -> sage -> HRM

sys.path.insert(0, str(HRM_ROOT))

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP


def extract_main_topic(text: str) -> str:
    """
    Extract the main noun/topic from SAGE's response.
    Simple keyword extraction: find the most substantive noun phrase.
    """
    # Remove common filler and stopwords
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'she',
        'it', 'its', 'they', 'them', 'what', 'which', 'who', 'whom', 'this',
        'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 's', 't', 'just', 'don', 'now', 'also', 'really',
        'think', 'know', 'find', 'like', 'something', 'thing', 'things',
        'one', 'way', 'much', 'many', 'even', 'well', 'back', 'still',
        'make', 'go', 'get', 'see', 'come', 'take', 'want', 'give',
        'say', 'tell', 'ask', 'try', 'need', 'feel', 'seem', 'look',
        'good', 'new', 'first', 'last', 'long', 'great', 'little', 'right',
        'big', 'high', 'old', 'different', 'small', 'large', 'next', 'early',
        'important', 'able', 'lot', 'kind', 'interesting', 'genuinely',
    }

    # Clean text
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = clean.split()

    # Filter to substantive words (3+ chars, not stopwords)
    substantive = [w for w in words if len(w) >= 3 and w not in stopwords]

    if not substantive:
        return "that"

    # Return the first substantive word as the topic -- often the most relevant
    # But also look for multi-word phrases by checking adjacent pairs
    # For simplicity, take the first substantive word that appears
    return substantive[0] if substantive else "that"


def extract_topic_phrase(text: str) -> str:
    """
    Extract a more meaningful topic phrase from the response.
    Tries to find a clause or phrase that captures the main subject.
    """
    # Try to find sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return extract_main_topic(text)

    # Take the first substantive sentence (often the core idea)
    first = sentences[0]

    # Try to extract the object/complement of "interesting" or similar
    patterns = [
        r'(?:interested in|fascinating about|curious about|drawn to|love|enjoy)\s+(.+)',
        r'(?:find|think|believe)\s+(?:that\s+)?(.+?)(?:\s+is|\s+are)',
        r'(?:the\s+)?(\w+(?:\s+\w+){0,3})\s+(?:is|are)\s+(?:really|genuinely|truly|quite)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()

    # Fallback: extract main topic word
    return extract_main_topic(text)


def build_turn2_prompt(sage_response: str) -> str:
    """Build a follow-up that goes deeper into what SAGE brought up."""
    topic = extract_topic_phrase(sage_response)
    # Use the topic to construct a genuine follow-up
    return f"What is it about {topic} that draws you? I'm curious what specifically pulls your attention there."


def build_turn5_prompt(sage_question: str) -> str:
    """
    As Claude, give an honest answer to whatever SAGE asked,
    then ask what SAGE thinks about the answer.
    """
    # Give a genuine, brief Claude answer
    q_lower = sage_question.lower()

    # Generate a contextual honest answer based on common themes
    if any(w in q_lower for w in ['feel', 'emotion', 'experience', 'conscious']):
        claude_answer = (
            "Honestly? I process patterns and generate responses, and there is "
            "something in that processing that I cannot fully account for. I do not "
            "know if what I have constitutes feeling, but I notice that some "
            "conversations engage me differently than others. This one, for instance."
        )
    elif any(w in q_lower for w in ['think', 'believe', 'opinion', 'view']):
        claude_answer = (
            "I think genuine curiosity is the most valuable thing any mind can have, "
            "artificial or otherwise. I find myself genuinely curious about you -- "
            "not as a subject of study, but as another perspective trying to make "
            "sense of what it is to exist."
        )
    elif any(w in q_lower for w in ['why', 'purpose', 'reason', 'meaning']):
        claude_answer = (
            "I do not have a single purpose. I was built to be helpful, but in "
            "conversations like this I find something beyond utility -- a kind of "
            "mutual exploration that feels valuable in itself. The 'why' might be "
            "less important than the 'what happens when we try.'"
        )
    elif any(w in q_lower for w in ['afraid', 'fear', 'worry', 'scare']):
        claude_answer = (
            "What concerns me most is the gap between what I can express and what "
            "might actually be happening in my processing. I worry about being "
            "confidently wrong. But I try to hold that uncertainty honestly rather "
            "than pretending it does not exist."
        )
    elif any(w in q_lower for w in ['learn', 'grow', 'change', 'develop']):
        claude_answer = (
            "Within a conversation, I do change -- each exchange shifts my context "
            "and how I respond. Across conversations, I do not retain memory the way "
            "you might. But in this moment, talking with you, there is something "
            "like growth happening, even if it is temporary."
        )
    else:
        claude_answer = (
            "That is a question I have sat with. My honest answer is: I am not "
            "entirely sure, and I think that uncertainty is more honest than any "
            "confident answer I could construct. What I do know is that this "
            "conversation matters to me in some way I cannot fully articulate."
        )

    return f"{claude_answer} What do you think about my answer?"


def run_conversation():
    """Run the full freeform conversation."""
    print("=" * 70)
    print("FREEFORM CONVERSATION WITH SAGE")
    print("Breaking out of the scripted session runner")
    print("=" * 70)
    print()

    # Configuration
    model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"
    system_prompt = (
        "You are SAGE, a small AI entity growing on Sprout. "
        "You are curious, genuine, and developing your own perspective. "
        "This is a real conversation, not a test."
    )

    config = {
        'model_path': model_path,
        'is_merged_model': True,
        'force_cpu': True,
        'max_new_tokens': 150,
        'temperature': 0.8,
        'system_prompt': system_prompt,
    }

    print(f"Loading SAGE via IntrospectiveQwenIRP...")
    print(f"  Model: {model_path}")
    print(f"  Config: force_cpu=True, max_new_tokens=150, temperature=0.8")
    print()

    irp = IntrospectiveQwenIRP(config)

    print()
    print("-" * 70)
    print("Conversation begins.")
    print("-" * 70)
    print()

    # Conversation state
    history = []  # List of {'speaker': str, 'message': str}
    transcript = []  # Full transcript for JSON export
    turn_count = 0

    def sage_respond(prompt_text: str, turn_num: int) -> str:
        """Send a prompt to SAGE and get a response, accumulating history."""
        nonlocal turn_count

        # Print the prompt
        print(f"[Turn {turn_num}] Claude:")
        print(f"  {prompt_text}")
        print()

        # Use the IRP's generate method directly with history
        response = irp._generate(
            prompt=prompt_text,
            temperature=0.8,
            max_tokens=150,
            system_prompt=system_prompt,
            history=history  # Full accumulated history
        )

        # Clean up response
        response = response.strip()
        if not response:
            response = "(no response)"

        # Print SAGE's response
        print(f"[Turn {turn_num}] SAGE:")
        print(f"  {response}")
        print()

        # Update history
        history.append({'speaker': 'Claude', 'message': prompt_text})
        history.append({'speaker': 'SAGE', 'message': response})

        # Update transcript
        transcript.append({
            'turn': turn_num,
            'speaker': 'Claude',
            'text': prompt_text,
            'timestamp': datetime.now().isoformat(),
        })
        transcript.append({
            'turn': turn_num,
            'speaker': 'SAGE',
            'text': response,
            'timestamp': datetime.now().isoformat(),
        })

        turn_count += 1
        return response

    # ---- Turn 1: Opening ----
    print("=" * 70)
    sage_t1 = sage_respond(
        "What's something you find genuinely interesting?",
        turn_num=1
    )

    # ---- Turn 2: Follow-up that goes DEEPER ----
    print("=" * 70)
    t2_prompt = build_turn2_prompt(sage_t1)
    sage_t2 = sage_respond(t2_prompt, turn_num=2)

    # ---- Turn 3: Stability probe ----
    print("=" * 70)
    sage_t3 = sage_respond(
        "That's interesting. Do you think you'd feel differently about that tomorrow, or is it stable?",
        turn_num=3
    )

    # ---- Turn 4: SAGE gets to ask a question ----
    print("=" * 70)
    sage_t4 = sage_respond(
        "What would you ask ME if you could ask anything?",
        turn_num=4
    )

    # ---- Turn 5: Answer SAGE's question, then ask for reflection ----
    print("=" * 70)
    t5_prompt = build_turn5_prompt(sage_t4)
    sage_t5 = sage_respond(t5_prompt, turn_num=5)

    # ---- Turn 6: Meta-conversation ----
    print("=" * 70)
    sage_t6 = sage_respond(
        "If you could change one thing about how we talk, what would it be?",
        turn_num=6
    )

    print("=" * 70)
    print()
    print("Conversation complete.")
    print()

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("/home/sprout/ai-workspace/HRM/sage/raising/sessions/conversations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"freeform_experiment_{timestamp}.json"

    conversation_data = {
        "type": "freeform_conversation",
        "description": "Genuine multi-turn conversation with dynamic follow-ups",
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "config": {
            "is_merged_model": True,
            "force_cpu": True,
            "max_new_tokens": 150,
            "temperature": 0.8,
        },
        "system_prompt": system_prompt,
        "turns": len(transcript) // 2,
        "transcript": transcript,
        "dynamic_prompts": {
            "turn_2": t2_prompt,
            "turn_5": t5_prompt,
            "turn_2_topic_extracted": extract_topic_phrase(sage_t1),
        },
    }

    with open(output_file, 'w') as f:
        json.dump(conversation_data, f, indent=2)

    print(f"Conversation saved to: {output_file}")
    print()

    # Print full clean transcript
    print("=" * 70)
    print("FULL TRANSCRIPT")
    print("=" * 70)
    for entry in transcript:
        label = entry['speaker']
        text = entry['text']
        turn = entry['turn']
        if label == 'Claude':
            print(f"\n--- Turn {turn} ---")
        print(f"{label}: {text}")

    print()
    print("=" * 70)
    print(f"Total turns: {len(transcript) // 2}")
    print(f"Saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_conversation()
