"""
SAGE CLI Client — talk to a running SAGE daemon from the command line.

Usage:
    # Single message (localhost)
    python3 -m sage.gateway.cli_client "Hello SAGE"

    # Single message (remote)
    python3 -m sage.gateway.cli_client --host 10.0.0.36 --port 8750 "Hello SAGE"

    # Interactive conversation
    python3 -m sage.gateway.cli_client --interactive

    # Health check
    python3 -m sage.gateway.cli_client --health

    # Full status
    python3 -m sage.gateway.cli_client --status

Version: 1.0 (2026-02-19)
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import uuid


def make_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def fetch_json(url: str, data: dict = None, timeout: float = 30) -> dict:
    """Send a request and return parsed JSON response."""
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={'Content-Type': 'application/json'},
        )
    else:
        req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ''
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return {'error': f"HTTP {e.code}: {e.reason}", 'body': body}
    except urllib.error.URLError as e:
        return {'error': f"Connection failed: {e.reason}"}
    except Exception as e:
        return {'error': str(e)}


def cmd_health(host: str, port: int):
    """Check SAGE daemon health."""
    url = make_url(host, port, '/health')
    result = fetch_json(url, timeout=5)

    if 'error' in result:
        print(f"[OFFLINE] SAGE at {host}:{port} — {result['error']}")
        return False

    state = result.get('metabolic_state', '?')
    atp = result.get('atp_level', '?')
    machine = result.get('machine', '?')
    cycles = result.get('cycle_count', '?')

    print(f"[ALIVE] SAGE at {host}:{port}")
    print(f"  Machine:   {machine}")
    print(f"  State:     {state}")
    print(f"  ATP:       {atp}")
    print(f"  Cycles:    {cycles}")
    return True


def cmd_status(host: str, port: int):
    """Get full daemon status."""
    url = make_url(host, port, '/status')
    result = fetch_json(url, timeout=5)

    if 'error' in result:
        print(f"[ERROR] {result['error']}")
        return

    print(json.dumps(result, indent=2))


def cmd_chat(host: str, port: int, message: str, sender: str,
             conversation_id: str = None, max_wait: int = 60):
    """Send a single message and print the response."""
    url = make_url(host, port, '/chat')
    data = {
        'sender': sender,
        'message': message,
        'max_wait_seconds': max_wait,
    }
    if conversation_id:
        data['conversation_id'] = conversation_id

    t0 = time.time()
    result = fetch_json(url, data=data, timeout=max_wait + 5)
    elapsed = time.time() - t0

    if 'error' in result:
        print(f"[ERROR] {result['error']}")
        if result.get('body'):
            print(f"  Body: {result['body']}")
        return result

    if result.get('status') == 'dreaming':
        print(f"[DREAMING] {result.get('message', 'SAGE is dreaming')}")
        return result

    response = result.get('response', '')
    state = result.get('metabolic_state', '?')
    atp = result.get('atp_remaining', '?')
    conv_id = result.get('conversation_id', '')

    print(f"\nSAGE [{state} | ATP {atp}]: {response}")
    print(f"  ({elapsed:.1f}s, conv: {conv_id[:12]}...)" if conv_id else f"  ({elapsed:.1f}s)")

    return result


def cmd_interactive(host: str, port: int, sender: str, max_wait: int = 60):
    """Interactive conversation with SAGE."""
    conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

    # Check health first
    if not cmd_health(host, port):
        print("\nSAGE is not reachable. Start the daemon first.")
        return

    print(f"\nInteractive conversation started (conv: {conversation_id[:16]}...)")
    print("Type 'quit' or 'exit' to end. Empty line to skip.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nConversation ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'bye'):
            print("Conversation ended.")
            break

        result = cmd_chat(host, port, user_input, sender,
                         conversation_id=conversation_id, max_wait=max_wait)

        # Check if SAGE is dreaming — offer to wait
        if isinstance(result, dict) and result.get('status') == 'dreaming':
            print("(Message queued. SAGE will process it when it wakes.)")

        print()  # blank line between turns


def main():
    parser = argparse.ArgumentParser(
        description='Talk to a running SAGE daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What's on your mind?"                  # Single message (localhost)
  %(prog)s --host 10.0.0.36 "Hello Sprout"         # Message to remote SAGE
  %(prog)s --interactive                            # Multi-turn conversation
  %(prog)s --health                                 # Check if SAGE is alive
  %(prog)s --status                                 # Full daemon status
  %(prog)s --host 10.0.0.36 --health                # Remote health check
""")

    parser.add_argument('message', nargs='?', default=None,
                       help='Message to send to SAGE')
    parser.add_argument('--host', default='127.0.0.1',
                       help='SAGE daemon host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8750,
                       help='SAGE daemon port (default: 8750)')
    parser.add_argument('--sender', default=None,
                       help='Sender identity (default: auto-detect)')
    parser.add_argument('--conversation-id', default=None,
                       help='Conversation ID for continuity')
    parser.add_argument('--max-wait', type=int, default=60,
                       help='Max seconds to wait for response (default: 60)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive multi-turn conversation')
    parser.add_argument('--health', action='store_true',
                       help='Check daemon health')
    parser.add_argument('--status', action='store_true',
                       help='Get full daemon status')

    args = parser.parse_args()

    # Auto-detect sender
    if args.sender is None:
        import socket
        hostname = socket.gethostname().lower()
        args.sender = f"cli@{hostname}"

    # Dispatch command
    if args.health:
        cmd_health(args.host, args.port)
    elif args.status:
        cmd_status(args.host, args.port)
    elif args.interactive:
        cmd_interactive(args.host, args.port, args.sender, args.max_wait)
    elif args.message:
        cmd_chat(args.host, args.port, args.message, args.sender,
                conversation_id=args.conversation_id, max_wait=args.max_wait)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
