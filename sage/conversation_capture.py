#!/usr/bin/env python3
"""
Real-time conversation capture - monitors SAGE conversation and logs to clean file
"""
import sys
import re
import time
from collections import deque

# Keep track of last seen conversation
last_user = None
last_sage = None
conversation_count = 0

print("="*80)
print("SAGE CONVERSATION CAPTURE")
print("="*80)
print("\nListening for conversations...")
print()

for line in sys.stdin:
    # Extract User input
    if '👤 User:' in line:
        user_match = re.search(r'👤 User: (.+)', line)
        if user_match:
            text = user_match.group(1).strip()
            if text not in ['(waiting for speech)', '(no response yet)']:
                if text != last_user:
                    last_user = text

    # Extract SAGE response
    elif '🤖 SAGE:' in line:
        sage_match = re.search(r'🤖 SAGE: (.+)', line)
        if sage_match:
            text = sage_match.group(1).strip()
            if text not in ['(waiting for speech)', '(no response yet)']:
                if text != last_sage and last_user:
                    last_sage = text
                    conversation_count += 1

                    # Print conversation
                    print(f"\n[{conversation_count}] {time.strftime('%H:%M:%S')}")
                    print(f"👤 YOU:  {last_user}")
                    print(f"🤖 SAGE: {last_sage}")

                    # Reset
                    last_user = None
                    last_sage = None

    # Extract path information
    elif '🔀 PATH:' in line:
        path_match = re.search(r'🔀 PATH: (.*)', line)
        if path_match:
            print(f"   {path_match.group(1)}")

    # Extract LLM latency
    elif 'LLM latency:' in line:
        latency_match = re.search(r'LLM latency: (.+)', line)
        if latency_match:
            print(f"   LLM latency: {latency_match.group(1)}")

    # Extract learned patterns
    elif 'NEW PATTERN LEARNED' in line:
        print(f"   📚 NEW PATTERN LEARNED!")
