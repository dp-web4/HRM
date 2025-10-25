#!/usr/bin/env python3
"""
Deduplicate conversation log - extracts unique conversation pairs with full responses
"""
import re
import sys

def parse_conversations(log_file):
    """Parse log file and extract unique conversation exchanges"""

    conversations = []
    current_entry = {}

    with open(log_file, 'r') as f:
        content = f.read()

    # Split by conversation number markers
    entries = re.findall(r'\[(\d+)\] (\d+:\d+:\d+)\n👤 YOU:  (.+?)\n🤖 SAGE: (.+?)(?:\n   🧠 (\w+)|🔀 PATH: ⚡ FAST|🔀 PATH: 🧠 SLOW)',
                        content, re.DOTALL)

    seen = set()
    unique_conversations = []

    for entry in entries:
        conv_num, timestamp, user_text, sage_text, path = entry

        # Create unique key from user + sage text
        key = (user_text.strip(), sage_text.strip()[:100])  # Use first 100 chars of response as key

        if key not in seen:
            seen.add(key)
            unique_conversations.append({
                'num': conv_num,
                'time': timestamp,
                'user': user_text.strip(),
                'sage': sage_text.strip(),
                'path': path if path else 'FAST'
            })

    return unique_conversations

def main():
    log_file = '/home/sprout/sage_conversations_clean.log'

    print("="*80)
    print("SAGE CONVERSATION TRANSCRIPT (Deduplicated)")
    print("="*80)
    print()

    conversations = parse_conversations(log_file)

    for i, conv in enumerate(conversations, 1):
        print(f"[{i}] {conv['time']}")
        print(f"👤 YOU:  {conv['user']}")
        print(f"🤖 SAGE: {conv['sage']}")
        print(f"   Path: {'⚡ FAST' if conv['path'] == 'FAST' else '🧠 SLOW (LLM)'}")
        print()

    print("="*80)
    print(f"Total unique exchanges: {len(conversations)}")
    print("="*80)

if __name__ == '__main__':
    main()
