"""Tool-use and retrieval interfaces for Prothynesis."""

from __future__ import annotations

import json
import time
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum


class ToolType(Enum):
    """Types of tools."""
    CALCULATOR = "calculator"
    PYTHON_EXECUTION = "python_execution"
    COMPILER = "compiler"
    TEST_RUNNER = "test_runner"
    SYMBOLIC_SOLVER = "symbolic_solver"
    RETRIEVER = "retriever"
    FILE_ANALYZER = "file_analyzer"
    SANDBOX = "sandbox"


@dataclass
class ToolCall:
    """A tool call request."""
    tool_id: str
    tool_type: str
    arguments: Dict[str, Any]
    call_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.call_id:
            self.call_id = f"call_{int(self.timestamp * 1000)}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    call_id: str
    tool_id: str
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseTool:
    """Base class for all tools."""

    def __init__(self, tool_id: str, tool_type: str, config: Optional[dict] = None):
        self.tool_id = tool_id
        self.tool_type = tool_type
        self.config = config or {}
        self.call_count = 0
        self.error_count = 0

    def can_handle(self, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this tool can handle the given task."""
        return False

    def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute the tool with given arguments."""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "tool_id": self.tool_id,
            "tool_type": self.tool_type,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.call_count),
        }


class CalculatorTool(BaseTool):
    """Simple calculator tool."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(tool_id="calculator", tool_type=ToolType.CALCULATOR.value, config=config)

    def can_handle(self, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
        task_lower = task.lower()
        return any(kw in task_lower for kw in ["calculate", "compute", "sum", "multiply", "divide", "math"])

    def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        self.call_count += 1
        try:
            expression = arguments.get("expression", "")
            result = eval(expression, {"__builtins__": {}}, {})
            return ToolResult(
                call_id="",
                tool_id=self.tool_id,
                success=True,
                output=str(result),
                execution_time_ms=0.0,
            )
        except Exception as e:
            self.error_count += 1
            return ToolResult(
                call_id="",
                tool_id=self.tool_id,
                success=False,
                output="",
                error=str(e),
            )


class PythonExecutionTool(BaseTool):
    """Safe Python code execution tool."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(tool_id="python_exec", tool_type=ToolType.PYTHON_EXECUTION.value, config=config)
        self.timeout = config.get("timeout", 10) if config else 10

    def can_handle(self, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
        task_lower = task.lower()
        return any(kw in task_lower for kw in ["python", "code", "script", "execute"])

    def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        self.call_count += 1
        code = arguments.get("code", "")

        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(code)
                temp_path = f.name

            start = time.time()
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = (time.time() - start) * 1000

            Path(temp_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return ToolResult(
                    call_id="",
                    tool_id=self.tool_id,
                    success=True,
                    output=result.stdout,
                    execution_time_ms=elapsed,
                )
            else:
                self.error_count += 1
                return ToolResult(
                    call_id="",
                    tool_id=self.tool_id,
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time_ms=elapsed,
                )
        except subprocess.TimeoutExpired:
            Path(temp_path).unlink(missing_ok=True)
            self.error_count += 1
            return ToolResult(
                call_id="",
                tool_id=self.tool_id,
                success=False,
                output="",
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            self.error_count += 1
            return ToolResult(
                call_id="",
                tool_id=self.tool_id,
                success=False,
                output="",
                error=str(e),
            )


class RetrieverTool(BaseTool):
    """Base retriever tool for RAG."""

    def __init__(self, tool_id: str = "retriever", config: Optional[dict] = None):
        super().__init__(tool_id=tool_id, tool_type=ToolType.RETRIEVER.value, config=config)
        self.documents: List[Dict[str, Any]] = []
        self.index: Dict[str, List[int]] = {}

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents for retrieval."""
        self.documents = documents
        self.index = {}
        for i, doc in enumerate(documents):
            text = doc.get("text", "").lower()
            for word in text.split():
                self.index.setdefault(word, []).append(i)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        if not self.documents:
            return []

        query_words = query.lower().split()
        scores: Dict[int, float] = {}

        for word in query_words:
            for doc_idx in self.index.get(word, []):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.documents[i] for i, _ in ranked[:top_k]]

    def can_handle(self, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
        return True

    def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        self.call_count += 1
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        results = self.retrieve(query, top_k)

        return ToolResult(
            call_id="",
            tool_id=self.tool_id,
            success=True,
            output=json.dumps(results),
            metadata={"count": len(results)},
        )


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.call_history: List[tuple[ToolCall, ToolResult]] = []

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> Optional[BaseTool]:
        """Get a tool by ID."""
        return self.tools.get(tool_id)

    def select_tool(self, task: str, context: Optional[Dict[str, Any]] = None) -> Optional[BaseTool]:
        """Select the best tool for a task."""
        best_tool = None
        best_score = 0.0

        for tool in self.tools.values():
            if tool.can_handle(task, context):
                score = 1.0 - tool.error_count / max(1, tool.call_count)
                if score > best_score:
                    best_score = score
                    best_tool = tool

        return best_tool

    def execute(self, tool_call: ToolCall, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute a tool call."""
        tool = self.tools.get(tool_call.tool_id)
        if not tool:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_id=tool_call.tool_id,
                success=False,
                output="",
                error=f"Tool not found: {tool_call.tool_id}",
            )

        result = tool.execute(tool_call.arguments, context)
        result.call_id = tool_call.call_id
        self.call_history.append((tool_call, result))
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "tool_count": len(self.tools),
            "total_calls": len(self.call_history),
            "tools": {tid: tool.get_stats() for tid, tool in self.tools.items()},
        }


def create_default_tool_registry() -> ToolRegistry:
    """Create a registry with default tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(PythonExecutionTool())
    registry.register(RetrieverTool())
    return registry
