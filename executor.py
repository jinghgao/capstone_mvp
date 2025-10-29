# executor.py - Tool Execution Layer
# Responsibility: Execute tool calls in sequence and collect evidence
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from rag import kb_retrieve, sop_extract
from sql_tool import sql_query_rag

# Import CV tool with fallback
try:
    from cv_tool import cv_assess_rag
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("[WARN] cv_tool not available, cv_assess_rag will be disabled")


# ========== TOOL REGISTRY ==========
TOOL_REGISTRY = {
    "kb_retrieve": kb_retrieve,
    "sop_extract": sop_extract,
    "sql_query_rag": sql_query_rag,
}
if CV_AVAILABLE:
    TOOL_REGISTRY["cv_assess_rag"] = cv_assess_rag


# ========== EXECUTION STATE ==========
class ExecutionState:
    """Tracks execution progress and results"""

    def __init__(self, slots: Dict[str, Any], status: str = "OK", message: Optional[str] = None,
                 clarifications: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.slots = slots or {}
        self.evidence = {
            "kb_hits": [],
            "sop": {},
            "sql": {},
            "cv": {},
            "support": []
        }
        self.logs: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.status = status
        self.message = message
        self.clarifications = clarifications or []
        self.metadata = metadata or {}

    def add_log(
        self,
        tool: str,
        args_keys: List[str],
        elapsed_ms: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Add execution log entry"""
        self.logs.append({
            "tool": tool,
            "args_redacted": args_keys,
            "elapsed_ms": elapsed_ms,
            "ok": success,
            "err": error
        })
        if not success and error:
            self.errors.append(f"{tool}: {error}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status,                     # "OK" | "NEEDS_CLARIFICATION" | "UNSUPPORTED" | "ERROR"
            "message": self.message,                   # Optional user-facing message
            "clarifications": self.clarifications,     # Ask-back prompts when params missing
            "slots": self.slots,
            "evidence": self.evidence,
            "logs": self.logs,
            "errors": self.errors,
            "metadata": self.metadata,
            "success": (len(self.errors) == 0) and (self.status == "OK")
        }


# ========== TOOL EXECUTOR ==========
class ToolExecutor:
    """
    Executes a sequence of tool calls and accumulates evidence
    """

    def __init__(self):
        self.tool_registry = TOOL_REGISTRY

    def execute(
        self,
        tool_chain: List[Dict[str, Any]],
        slots: Dict[str, Any],
        status: str = "OK",
        message: Optional[str] = None,
        clarifications: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionState:
        """
        Execute tool chain in sequence

        Short-circuits:
          - If status == "UNSUPPORTED" -> return with message (no execution)
          - If clarifications not empty -> return with clarifications (no execution)
          - If tool_chain is empty and status == "OK" -> return empty success shell
        """
        # Short-circuit: unsupported
        if status == "UNSUPPORTED":
            return ExecutionState(
                slots=slots,
                status="UNSUPPORTED",
                message=message or "This question is not supported yet",
                clarifications=[],
                metadata=metadata,
            )

        # Short-circuit: needs clarification
        if clarifications:
            return ExecutionState(
                slots=slots,
                status="NEEDS_CLARIFICATION",
                message=None,
                clarifications=clarifications,
                metadata=metadata,
            )

        state = ExecutionState(slots=slots, status="OK", message=None, metadata=metadata)

        if not tool_chain:
            # Nothing to execute; return clean OK state (useful for pure-clarification flows)
            return state

        print(f"[Executor] Starting execution of {len(tool_chain)} tools")

        for step_idx, step in enumerate(tool_chain):
            tool_name = step.get("tool")
            args = step.get("args", {}) or {}

            print(f"[Executor] Step {step_idx + 1}/{len(tool_chain)}: {tool_name}")

            # Execute tool
            success, error, elapsed_ms = self._execute_tool(
                tool_name,
                args,
                state
            )

            # Log execution
            state.add_log(
                tool=tool_name,
                args_keys=list(args.keys()),
                elapsed_ms=elapsed_ms,
                success=success,
                error=error
            )

            # Stop on critical errors (optional)
            if not success and error and "CRITICAL" in error:
                print(f"[Executor] Critical error, stopping execution")
                state.status = "ERROR"
                state.message = "Execution aborted due to critical error."
                break

        print(f"[Executor] Execution complete: {len(state.logs)} tools executed")
        print(f"[Executor] Success: {len(state.errors) == 0}")
        return state

    def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: ExecutionState
    ) -> Tuple[bool, Optional[str], int]:
        """
        Execute a single tool and update state

        Returns:
            (success, error_message, elapsed_ms)
        """
        start_time = time.time()

        try:
            if tool_name == "kb_retrieve":
                return self._exec_kb_retrieve(args, state, start_time)

            elif tool_name == "sop_extract":
                return self._exec_sop_extract(args, state, start_time)

            elif tool_name == "sql_query_rag":
                return self._exec_sql_query(args, state, start_time)

            elif tool_name == "cv_assess_rag":
                return self._exec_cv_assess(args, state, start_time)

            else:
                elapsed_ms = int((time.time() - start_time) * 1000)
                error = f"Unknown tool: {tool_name}"
                print(f"[Executor ERROR] {error}")
                return False, error, elapsed_ms

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error = f"Exception: {str(e)}"
            print(f"[Executor ERROR] Tool '{tool_name}' failed: {error}")
            return False, error, elapsed_ms

    # ========== TOOL HANDLERS ==========
    def _exec_kb_retrieve(
        self,
        args: Dict[str, Any],
        state: ExecutionState,
        start_time: float
    ) -> Tuple[bool, Optional[str], int]:
        """Execute knowledge base retrieval"""
        result = self.tool_registry["kb_retrieve"](**args)
        state.evidence["kb_hits"] = result.get("hits", [])
        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"[Executor] kb_retrieve: {len(state.evidence['kb_hits'])} hits retrieved")
        return True, None, elapsed_ms

    def _exec_sop_extract(
        self,
        args: Dict[str, Any],
        state: ExecutionState,
        start_time: float
    ) -> Tuple[bool, Optional[str], int]:
        """Extract SOP from previously retrieved documents"""
        snippets = [hit.get("text", "") for hit in state.evidence.get("kb_hits", [])]
        if not snippets:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return False, "No documents to extract SOP from", elapsed_ms

        if "schema" not in args:
            args["schema"] = None

        result = self.tool_registry["sop_extract"](snippets=snippets, **args)
        state.evidence["sop"] = result or {}
        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"[Executor] sop_extract: extracted {len(state.evidence['sop'])} SOP fields")
        return True, None, elapsed_ms

    def _exec_sql_query(
        self,
        args: Dict[str, Any],
        state: ExecutionState,
        start_time: float
    ) -> Tuple[bool, Optional[str], int]:
        """Execute SQL query"""
        result = self.tool_registry["sql_query_rag"](**args)

        # Store SQL results
        state.evidence["sql"] = {k: v for k, v in result.items() if k in ("rows", "rowcount", "elapsed_ms")}

        # Store support documents (if any)
        if "support" in result:
            state.evidence["support"] = result["support"]

        elapsed_ms = int((time.time() - start_time) * 1000)
        row_count = state.evidence["sql"].get("rowcount", 0)
        print(f"[Executor] sql_query_rag: {row_count} rows returned")
        return True, None, elapsed_ms

    def _exec_cv_assess(
        self,
        args: Dict[str, Any],
        state: ExecutionState,
        start_time: float
    ) -> Tuple[bool, Optional[str], int]:
        """Execute computer vision assessment"""
        if not CV_AVAILABLE:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return False, "cv_tool not available", elapsed_ms

        result = self.tool_registry["cv_assess_rag"](**args)

        if "cv" in result:
            state.evidence["cv"] = result["cv"]
        if "support" in result:
            state.evidence["support"] = result["support"]

        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"[Executor] cv_assess_rag: analysis complete")
        return True, None, elapsed_ms


# ========== CONVENIENCE FUNCTIONS ==========
def execute_plan(
    tool_chain: List[Dict[str, Any]],
    slots: Dict[str, Any],
    clarifications: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a tool chain (backwards compatible interface).
    If clarifications are provided -> short-circuit and return them.
    If metadata.status == "UNSUPPORTED" -> short-circuit with message.
    """
    status = (metadata or {}).get("status", "OK")
    message = None
    if status == "UNSUPPORTED":
        message = "This question is not supported yet"

    executor = ToolExecutor()
    state = executor.execute(
        tool_chain=tool_chain,
        slots=slots,
        status=status,
        message=message,
        clarifications=clarifications,
        metadata=metadata,
    )
    return state.to_dict()


def execute_from_plan(
    plan: Union[Dict[str, Any], Any],
    slots: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute directly from a planner plan (plan.to_dict()) plus slots.
    Expected dict fields: tool_chain, clarifications, metadata.
    """
    if hasattr(plan, "to_dict"):
        plan = plan.to_dict()  # type: ignore

    tool_chain = plan.get("tool_chain", [])
    clarifications = plan.get("clarifications", []) or []
    metadata = plan.get("metadata", {}) or {}
    return execute_plan(tool_chain=tool_chain, slots=slots, clarifications=clarifications, metadata=metadata)