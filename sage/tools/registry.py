"""
Tool Registry — central registration and schema for SAGE tools.

Each tool is a ToolDefinition: name, description, parameters (JSON Schema),
a callable Python function, ATP cost, and policy level.

The registry provides lookup, schema export (for Ollama /api/chat and
prompt injection), and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional


@dataclass
class ToolCall:
    """A parsed tool invocation from LLM output."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'arguments': self.arguments}


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {'tool_name': self.tool_name, 'success': self.success}
        if self.success:
            d['result'] = self.result
        else:
            d['error'] = self.error
        return d

    def to_text(self) -> str:
        """Format for re-injection into LLM context."""
        if self.success:
            return f"[Tool {self.tool_name} result]: {self.result}"
        return f"[Tool {self.tool_name} error]: {self.error}"


@dataclass
class ToolDefinition:
    """
    A registered tool that SAGE can invoke.

    Attributes:
        name: Unique identifier (e.g. 'web_search')
        description: Human-readable description for prompt injection
        parameters: JSON Schema dict describing expected arguments
        fn: The Python callable to execute
        atp_cost: ATP cost per invocation (deducted from metabolic budget)
        policy_level: 'standard' | 'elevated' | 'dangerous'
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable
    atp_cost: float = 1.0
    policy_level: str = 'standard'  # standard | elevated | dangerous

    def to_ollama_tool(self) -> Dict[str, Any]:
        """Format for Ollama /api/chat tools parameter (T1 native)."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            }
        }

    def to_prompt_text(self) -> str:
        """Format for prompt injection (T2/T3)."""
        param_desc = []
        props = self.parameters.get('properties', {})
        required = self.parameters.get('required', [])
        for pname, pschema in props.items():
            req_mark = ' (required)' if pname in required else ''
            ptype = pschema.get('type', 'any')
            pdesc = pschema.get('description', '')
            param_desc.append(f"    - {pname} ({ptype}{req_mark}): {pdesc}")

        params_text = '\n'.join(param_desc) if param_desc else '    (no parameters)'
        return f"- {self.name}: {self.description}\n  Parameters:\n{params_text}"


class ToolRegistry:
    """
    Central registry of available tools.

    Tools are registered at daemon startup and made available to the
    consciousness loop based on policy level and metabolic state.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        """Register a tool definition."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, policy_level: Optional[str] = None) -> List[ToolDefinition]:
        """List all tools, optionally filtered by policy level."""
        tools = list(self._tools.values())
        if policy_level:
            tools = [t for t in tools if t.policy_level == policy_level]
        return tools

    @property
    def names(self) -> List[str]:
        """All registered tool names."""
        return list(self._tools.keys())

    def to_ollama_tools(self, policy_level: Optional[str] = None) -> List[Dict]:
        """Export all tools as Ollama /api/chat tool definitions."""
        return [t.to_ollama_tool() for t in self.list_tools(policy_level)]

    def to_prompt_block(self, policy_level: Optional[str] = None) -> str:
        """Export all tools as a text block for prompt injection."""
        tools = self.list_tools(policy_level)
        if not tools:
            return ''
        lines = ['Available tools:']
        for t in tools:
            lines.append(t.to_prompt_text())
        return '\n'.join(lines)

    def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Looks up the tool, validates it exists, calls fn(**arguments).
        Returns a ToolResult with success/failure and result/error.
        """
        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(
                tool_name=call.name,
                success=False,
                error=f"Unknown tool: {call.name}"
            )

        try:
            result = tool.fn(**call.arguments)
            return ToolResult(
                tool_name=call.name,
                success=True,
                result=result,
            )
        except Exception as e:
            return ToolResult(
                tool_name=call.name,
                success=False,
                error=str(e),
            )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ============================================================================
# Inline tests
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ToolRegistry — Inline Tests")
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

    # Test tool
    def dummy_tool(query: str = "test") -> str:
        return f"result for {query}"

    td = ToolDefinition(
        name='test_tool',
        description='A test tool',
        parameters={
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Search query'}
            },
            'required': ['query'],
        },
        fn=dummy_tool,
        atp_cost=0.5,
        policy_level='standard',
    )

    # Test ToolDefinition
    ollama_fmt = td.to_ollama_tool()
    check("ollama format has function", 'function' in ollama_fmt)
    check("ollama function name", ollama_fmt['function']['name'] == 'test_tool')

    prompt_text = td.to_prompt_text()
    check("prompt text has name", 'test_tool' in prompt_text)
    check("prompt text has description", 'A test tool' in prompt_text)

    # Test ToolRegistry
    reg = ToolRegistry()
    reg.register(td)
    check("registry has tool", 'test_tool' in reg)
    check("registry len", len(reg) == 1)
    check("registry get", reg.get('test_tool') is td)
    check("registry get missing", reg.get('nope') is None)

    # Test execution
    call = ToolCall(name='test_tool', arguments={'query': 'hello'})
    result = reg.execute(call)
    check("execute success", result.success)
    check("execute result", result.result == 'result for hello')
    check("execute to_text", 'result for hello' in result.to_text())

    # Test failed execution
    bad_call = ToolCall(name='nonexistent', arguments={})
    bad_result = reg.execute(bad_call)
    check("execute unknown tool fails", not bad_result.success)
    check("execute unknown tool error", 'Unknown tool' in bad_result.error)

    # Test listing
    check("list all tools", len(reg.list_tools()) == 1)
    check("list filtered", len(reg.list_tools('standard')) == 1)
    check("list filtered empty", len(reg.list_tools('elevated')) == 0)

    # Test ToolCall
    tc = ToolCall(name='foo', arguments={'x': 1})
    check("toolcall to_dict", tc.to_dict() == {'name': 'foo', 'arguments': {'x': 1}})

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        exit(1)
