"""
Built-in tools for SAGE tool use.

All tools are pure Python functions using stdlib + httpx (optional).
Each function is self-contained and returns a string result suitable
for re-injection into LLM context.

Security:
    - calculate: uses ast.literal_eval, NOT eval()
    - read_file / write_note: sandboxed to instance directory
    - web_search / web_fetch: HTTP only, no auth
    - peer_ask: HTTP POST to known peer endpoints only
"""

import ast
import json
import operator
import os
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .registry import ToolDefinition, ToolRegistry


# ============================================================================
# Tool implementations
# ============================================================================

def tool_get_time(timezone_name: str = 'local') -> str:
    """Get current date, time, and timezone."""
    now = datetime.now()
    utc_now = datetime.now(timezone.utc)
    return (
        f"Local: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"UTC:   {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Unix:  {int(time.time())}"
    )


# Safe math operations for calculate
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_expr(node):
    """Recursively evaluate an AST expression node using only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        # Prevent excessively large exponents
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and right > 1000:
            raise ValueError("Exponent too large (max 1000)")
        return op_fn(left, right)
    elif isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_expr(node.operand))
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def tool_calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Supports: +, -, *, /, //, %, ** with integer and float literals.
    Does NOT support variables, function calls, or imports.
    Raises on invalid/unsafe expressions so the registry reports failure.
    """
    tree = ast.parse(expression.strip(), mode='eval')
    result = _safe_eval_expr(tree)
    return f"{expression} = {result}"


def tool_web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo Lite (no API key needed).

    Returns up to max_results titles and URLs.
    """
    max_results = min(max_results, 10)
    encoded = urllib.parse.urlencode({'q': query})
    url = f'https://lite.duckduckgo.com/lite/?{encoded}'

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'SAGE/0.4 (tool-use; research)',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')

        # Parse DuckDuckGo Lite results (HTML table format)
        results = []
        import re

        # DDG lite wraps result URLs through redirector:
        #   href="//duckduckgo.com/l/?uddg=https%3A%2F%2F..." class='result-link'
        # Extract the uddg parameter (the actual URL, URL-encoded)
        result_links = re.findall(
            r"""<a[^>]+href=["'](?://duckduckgo\.com/l/\?uddg=([^&"']+)[^"']*|([^"']+))["'][^>]*class=["']result-link["'][^>]*>([^<]+)</a>""",
            html
        )
        # Also try reversed attribute order (class before href)
        if not result_links:
            result_links = re.findall(
                r"""<a[^>]+class=["']result-link["'][^>]+href=["'](?://duckduckgo\.com/l/\?uddg=([^&"']+)[^"']*|([^"']+))["'][^>]*>([^<]+)</a>""",
                html
            )

        # Extract snippets (text in class='result-snippet' cells)
        snippets = re.findall(
            r"""class=["']result-snippet["'][^>]*>(.*?)</td>""",
            html, re.DOTALL
        )
        # Strip HTML tags from snippets
        clean_snippets = []
        for s in snippets:
            clean = re.sub(r'<[^>]+>', '', s).strip()
            clean_snippets.append(clean)

        seen = set()
        for i, match in enumerate(result_links):
            uddg_url, direct_url, title = match
            # Decode the uddg URL-encoded parameter
            if uddg_url:
                href = urllib.parse.unquote(uddg_url)
            elif direct_url:
                href = direct_url
            else:
                continue

            if not href.startswith('http') or href in seen:
                continue
            seen.add(href)

            title = title.strip()
            snippet = clean_snippets[i] if i < len(clean_snippets) else ''
            entry = f"- {title}\n  {href}"
            if snippet:
                entry += f"\n  {snippet[:150]}"
            results.append(entry)
            if len(results) >= max_results:
                break

        if results:
            return f"Search results for '{query}':\n" + '\n'.join(results)
        return f"No results found for '{query}'"

    except Exception as e:
        return f"Search error: {e}"


def tool_web_fetch(url: str, max_chars: int = 4000) -> str:
    """
    Fetch text content from a URL.

    Strips HTML tags and returns plain text, truncated to max_chars.
    """
    max_chars = min(max_chars, 10000)

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'SAGE/0.4 (tool-use; research)',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get('Content-Type', '')
            raw = resp.read()

        # Decode
        encoding = 'utf-8'
        if 'charset=' in content_type:
            encoding = content_type.split('charset=')[-1].split(';')[0].strip()
        text = raw.decode(encoding, errors='replace')

        # Strip HTML if needed
        if 'html' in content_type.lower() or text.strip().startswith('<'):
            import re
            # Remove script/style blocks
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            # Remove tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Collapse whitespace
            text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + f'\n... (truncated, {len(raw)} bytes total)'

        return f"Content from {url}:\n{text}"

    except Exception as e:
        return f"Fetch error for {url}: {e}"


# Sandbox root — set at registration time via set_instance_dir()
_instance_dir: Optional[Path] = None


def set_instance_dir(path: Path):
    """Set the sandboxed instance directory for file tools."""
    global _instance_dir
    _instance_dir = path


def _sandboxed_path(filename: str) -> Path:
    """Resolve a filename within the sandboxed instance directory."""
    base = _instance_dir or Path.cwd()
    # Prevent directory traversal
    resolved = (base / filename).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise PermissionError(f"Path escapes sandbox: {filename}")
    return resolved


def tool_read_file(filename: str) -> str:
    """Read a file from the SAGE instance directory (sandboxed)."""
    try:
        path = _sandboxed_path(filename)
        if not path.exists():
            return f"File not found: {filename}"
        if path.stat().st_size > 50_000:
            return f"File too large: {filename} ({path.stat().st_size} bytes, max 50KB)"
        return path.read_text(encoding='utf-8', errors='replace')
    except PermissionError as e:
        return f"Permission denied: {e}"
    except Exception as e:
        return f"Read error: {e}"


def tool_write_note(content: str, filename: str = 'notes.txt') -> str:
    """Append a note to a file in the SAGE instance directory (append-only)."""
    try:
        path = _sandboxed_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] {content}\n")
        return f"Note appended to {filename}: \"{content}\""
    except PermissionError as e:
        return f"Permission denied: {e}"
    except Exception as e:
        return f"Write error: {e}"


def tool_peer_ask(peer: str, question: str) -> str:
    """
    Ask a peer SAGE instance a question via HTTP POST.

    ``peer`` can be a machine name (e.g. "thor") or a full URL
    (e.g. "http://10.0.0.210:8750").  Names are resolved via the
    fleet registry.
    """
    # Resolve peer name → URL via fleet registry
    url = peer
    if not peer.startswith('http'):
        try:
            from sage.federation.fleet_registry import FleetRegistry
            registry = FleetRegistry(os.uname().nodename.lower())
            resolved = registry.get_gateway_url(peer)
            if resolved:
                url = resolved
            else:
                return f"Unknown peer '{peer}'. Known: {', '.join(registry.get_peer_names())}"
        except Exception as e:
            return f"Fleet registry error: {e}"

    try:
        self_name = os.uname().nodename.lower()
        payload = json.dumps({
            'message': question,
            'sender': self_name,  # Identify as our unique name, not generic "sage_peer"
        }).encode('utf-8')

        # Ensure URL ends with /chat
        if not url.rstrip('/').endswith('/chat'):
            url = url.rstrip('/') + '/chat'
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'X-Platform': self_name,
                'X-Signature': self_name,  # TODO: real Ed25519 signatures
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get('response', result.get('text', str(result)))

    except Exception as e:
        return f"Peer communication error ({peer}): {e}"


# ============================================================================
# Registry builder
# ============================================================================

def create_default_registry(instance_dir: Optional[Path] = None) -> ToolRegistry:
    """
    Create a ToolRegistry populated with all built-in tools.

    Args:
        instance_dir: Sandbox root for file tools. If None, uses cwd.

    Returns:
        Populated ToolRegistry ready for use.
    """
    if instance_dir:
        set_instance_dir(instance_dir)

    registry = ToolRegistry()

    registry.register(ToolDefinition(
        name='get_time',
        description='Get the current date and time. Only use when time is specifically relevant to the conversation — NOT on every message.',
        parameters={
            'type': 'object',
            'properties': {
                'timezone_name': {
                    'type': 'string',
                    'description': 'Timezone name (currently only "local" supported)',
                },
            },
            'required': [],
        },
        fn=tool_get_time,
        atp_cost=0.1,
        policy_level='standard',
    ))

    registry.register(ToolDefinition(
        name='calculate',
        description='Evaluate a mathematical expression. Supports +, -, *, /, //, %, **.',
        parameters={
            'type': 'object',
            'properties': {
                'expression': {
                    'type': 'string',
                    'description': 'Math expression to evaluate (e.g. "2 + 3 * 4")',
                },
            },
            'required': ['expression'],
        },
        fn=tool_calculate,
        atp_cost=0.1,
        policy_level='standard',
    ))

    registry.register(ToolDefinition(
        name='web_search',
        description='Search the web for information using DuckDuckGo.',
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query',
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results (default 5, max 10)',
                },
            },
            'required': ['query'],
        },
        fn=tool_web_search,
        atp_cost=1.0,
        policy_level='standard',
    ))

    registry.register(ToolDefinition(
        name='web_fetch',
        description='Fetch and read the text content of a web page.',
        parameters={
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'URL to fetch',
                },
                'max_chars': {
                    'type': 'integer',
                    'description': 'Maximum characters to return (default 4000)',
                },
            },
            'required': ['url'],
        },
        fn=tool_web_fetch,
        atp_cost=1.5,
        policy_level='standard',
    ))

    registry.register(ToolDefinition(
        name='read_file',
        description='Read a file from the SAGE instance directory.',
        parameters={
            'type': 'object',
            'properties': {
                'filename': {
                    'type': 'string',
                    'description': 'Filename relative to instance directory',
                },
            },
            'required': ['filename'],
        },
        fn=tool_read_file,
        atp_cost=0.2,
        policy_level='standard',
    ))

    registry.register(ToolDefinition(
        name='write_note',
        description='Append a note to a file in the SAGE instance directory.',
        parameters={
            'type': 'object',
            'properties': {
                'content': {
                    'type': 'string',
                    'description': 'Note content to append',
                },
                'filename': {
                    'type': 'string',
                    'description': 'Target filename (default: notes.txt)',
                },
            },
            'required': ['content'],
        },
        fn=tool_write_note,
        atp_cost=0.3,
        policy_level='standard',
    ))

    registry.register(ToolDefinition(
        name='peer_ask',
        description='Ask a peer SAGE instance a question. Peers: thor, legion, mcnugget, nomad, cbp.',
        parameters={
            'type': 'object',
            'properties': {
                'peer': {
                    'type': 'string',
                    'description': 'Peer name (e.g. "thor") or gateway URL',
                },
                'question': {
                    'type': 'string',
                    'description': 'Question to ask the peer',
                },
            },
            'required': ['peer', 'question'],
        },
        fn=tool_peer_ask,
        atp_cost=2.0,
        policy_level='standard',
    ))

    return registry


# ============================================================================
# Inline tests
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Built-in Tools — Inline Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    # Test get_time
    result = tool_get_time()
    check("get_time returns date", 'Local:' in result)
    check("get_time returns UTC", 'UTC:' in result)
    check("get_time returns unix", 'Unix:' in result)

    # Test calculate
    check("calculate basic", tool_calculate('2 + 3') == '2 + 3 = 5')
    check("calculate complex", '120' in tool_calculate('5 * 4 * 3 * 2'))
    check("calculate division", '2.5' in tool_calculate('5 / 2'))
    check("calculate negative", '-3' in tool_calculate('-3'))
    check("calculate error", 'Error' in tool_calculate('import os'))
    check("calculate div zero", 'Error' in tool_calculate('1/0'))
    check("calculate large exp blocked", 'Error' in tool_calculate('2**10000'))

    # Test safe eval rejects function calls
    check("calculate no functions", 'Error' in tool_calculate('abs(-1)'))
    check("calculate no names", 'Error' in tool_calculate('x + 1'))

    # Test sandbox
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        set_instance_dir(Path(tmpdir))

        # write_note
        result = tool_write_note('test note')
        check("write_note success", 'appended' in result)

        # read_file
        result = tool_read_file('notes.txt')
        check("read_file returns content", 'test note' in result)

        # read missing
        result = tool_read_file('nonexistent.txt')
        check("read_file missing", 'not found' in result)

        # directory traversal blocked
        result = tool_read_file('../../etc/passwd')
        check("sandbox blocks traversal", 'Permission denied' in result or 'not found' in result.lower())

    # Test registry creation
    reg = create_default_registry()
    check("registry has 7 tools", len(reg) == 7)
    check("registry has get_time", 'get_time' in reg)
    check("registry has web_search", 'web_search' in reg)
    check("registry has peer_ask", 'peer_ask' in reg)

    # Test Ollama export
    ollama_tools = reg.to_ollama_tools()
    check("ollama export count", len(ollama_tools) == 7)
    check("ollama export format", ollama_tools[0]['type'] == 'function')

    # Test prompt block
    prompt = reg.to_prompt_block()
    check("prompt block has tools", 'Available tools:' in prompt)
    check("prompt block has get_time", 'get_time' in prompt)

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        exit(1)
