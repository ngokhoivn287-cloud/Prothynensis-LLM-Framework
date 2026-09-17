"""Tests for tool-use and retrieval interfaces."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_calculator_tool():
    """Test calculator tool."""
    from src.training.tools import CalculatorTool

    tool = CalculatorTool()
    assert tool.can_handle("calculate 2+2")
    assert tool.can_handle("compute sum")

    result = tool.execute({"expression": "2 + 3 * 4"})
    assert result.success is True
    assert result.output == "14"
    print("  PASS: Calculator tool works")


def test_python_execution_tool():
    """Test Python execution tool."""
    from src.training.tools import PythonExecutionTool

    tool = PythonExecutionTool(config={"timeout": 5})
    assert tool.can_handle("write a python script")

    result = tool.execute({"code": "print(42)"})
    assert result.success is True
    assert "42" in result.output
    print("  PASS: Python execution tool works")


def test_retriever_tool():
    """Test retriever tool."""
    from src.training.tools import RetrieverTool

    tool = RetrieverTool()
    docs = [
        {"text": "Python is a programming language", "source": "doc1"},
        {"text": "JavaScript runs in browsers", "source": "doc2"},
        {"text": "Python is used for data science", "source": "doc3"},
    ]
    tool.index_documents(docs)

    results = tool.retrieve("Python programming", top_k=2)
    assert len(results) == 2
    assert any("Python" in r["text"] for r in results)
    print("  PASS: Retriever tool works")


def test_tool_registry():
    """Test tool registry."""
    from src.training.tools import ToolRegistry, CalculatorTool, PythonExecutionTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(PythonExecutionTool())

    tool = registry.select_tool("calculate 5+5")
    assert tool is not None
    assert tool.tool_id == "calculator"

    tool = registry.select_tool("run python code")
    assert tool is not None
    assert tool.tool_id == "python_exec"

    stats = registry.get_stats()
    assert stats["tool_count"] == 2
    print("  PASS: Tool registry works")


def test_tool_registry_execute():
    """Test tool execution through registry."""
    from src.training.tools import ToolRegistry, CalculatorTool, ToolCall

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    tool_call = ToolCall(tool_id="calculator", tool_type="calculator", arguments={"expression": "10 / 2"})
    result = registry.execute(tool_call)
    assert result.success is True
    assert result.output == "5.0"
    assert len(registry.call_history) == 1
    print("  PASS: Registry execution works")


def test_tool_call_creation():
    """Test tool call creation."""
    from src.training.tools import ToolCall

    call = ToolCall(tool_id="test", tool_type="test", arguments={"arg": "value"})
    assert call.call_id != ""
    data = call.to_dict()
    assert data["tool_id"] == "test"
    assert data["arguments"] == {"arg": "value"}
    print("  PASS: Tool call creation works")


def test_tool_result_creation():
    """Test tool result creation."""
    from src.training.tools import ToolResult

    result = ToolResult(
        call_id="call_1",
        tool_id="test",
        success=True,
        output="result",
        metadata={"key": "value"},
    )
    data = result.to_dict()
    assert data["success"] is True
    assert data["output"] == "result"
    print("  PASS: Tool result creation works")


def test_tool_stats():
    """Test tool statistics tracking."""
    from src.training.tools import CalculatorTool

    tool = CalculatorTool()
    tool.execute({"expression": "1+1"})
    tool.execute({"expression": "invalid"})

    stats = tool.get_stats()
    assert stats["call_count"] == 2
    assert stats["error_count"] == 1
    assert stats["error_rate"] == 0.5
    print("  PASS: Tool stats tracked")


def test_create_default_tool_registry():
    """Test default tool registry creation."""
    from src.training.tools import create_default_tool_registry

    registry = create_default_tool_registry()
    assert len(registry.tools) >= 3
    print("  PASS: Default tool registry created")


def test_tool_type_enum():
    """Test tool type enum."""
    from src.training.tools import ToolType

    assert ToolType.CALCULATOR.value == "calculator"
    assert ToolType.RETRIEVER.value == "retriever"
    print("  PASS: Tool types defined")


def run_all_tests():
    """Run all tool tests."""
    tests = [
        ("Calculator Tool", test_calculator_tool),
        ("Python Execution Tool", test_python_execution_tool),
        ("Retriever Tool", test_retriever_tool),
        ("Tool Registry", test_tool_registry),
        ("Registry Execute", test_tool_registry_execute),
        ("Tool Call Creation", test_tool_call_creation),
        ("Tool Result Creation", test_tool_result_creation),
        ("Tool Stats", test_tool_stats),
        ("Default Registry", test_create_default_tool_registry),
        ("Tool Type Enum", test_tool_type_enum),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
