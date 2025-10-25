# executor.py - Tool Execution Layer
# Responsibility: Execute tool calls in sequence and collect evidence
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

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
    
    def __init__(self, slots: Dict[str, Any]):
        self.slots = slots
        self.evidence = {
            "kb_hits": [],
            "sop": {},
            "sql": {},
            "cv": {},
            "support": []
        }
        self.logs = []
        self.errors = []
    
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
            "slots": self.slots,
            "evidence": self.evidence,
            "logs": self.logs,
            "errors": self.errors,
            "success": len(self.errors) == 0
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
        slots: Dict[str, Any]
    ) -> ExecutionState:
        """
        Execute tool chain in sequence
        
        Args:
            tool_chain: List of tool calls with args
            slots: User intent slots from NLU
        
        Returns:
            ExecutionState with evidence and logs
        """
        state = ExecutionState(slots)
        
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
            
            # Stop on critical errors (optional - you can continue on errors)
            if not success and error and "CRITICAL" in error:
                print(f"[Executor] Critical error, stopping execution")
                break
        
        print(f"[Executor] Execution complete: {len(state.logs)} tools executed")
        print(f"[Executor] Success: {len(state.errors) == 0}")
        
        return state
    
    def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: ExecutionState
    ) -> tuple[bool, Optional[str], int]:
        """
        Execute a single tool and update state
        
        Returns:
            (success, error_message, elapsed_ms)
        """
        start_time = time.time()
        
        try:
            # Route to appropriate handler
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
    ) -> tuple[bool, Optional[str], int]:
        """Execute knowledge base retrieval"""
        result = self.tool_registry["kb_retrieve"](**args)
        
        # Store results in state
        state.evidence["kb_hits"] = result.get("hits", [])
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"[Executor] kb_retrieve: {len(state.evidence['kb_hits'])} hits retrieved")
        
        return True, None, elapsed_ms
    
    def _exec_sop_extract(
        self,
        args: Dict[str, Any],
        state: ExecutionState,
        start_time: float
    ) -> tuple[bool, Optional[str], int]:
        """Extract SOP from previously retrieved documents"""
        # Get snippets from previous kb_retrieve
        snippets = [hit["text"] for hit in state.evidence["kb_hits"]]
        
        if not snippets:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return False, "No documents to extract SOP from", elapsed_ms
        
        # Add schema if not provided
        if "schema" not in args:
            args["schema"] = None
        
        result = self.tool_registry["sop_extract"](snippets=snippets, **args)
        
        # Store results in state
        state.evidence["sop"] = result
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"[Executor] sop_extract: extracted {len(result)} SOP fields")
        
        return True, None, elapsed_ms
    
    def _exec_sql_query(
        self,
        args: Dict[str, Any],
        state: ExecutionState,
        start_time: float
    ) -> tuple[bool, Optional[str], int]:
        """Execute SQL query"""
        result = self.tool_registry["sql_query_rag"](**args)
        
        # Store SQL results
        state.evidence["sql"] = {
            k: v for k, v in result.items()
            if k in ("rows", "rowcount", "elapsed_ms")
        }
        
        # Store support documents
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
    ) -> tuple[bool, Optional[str], int]:
        """Execute computer vision assessment"""
        if not CV_AVAILABLE:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return False, "cv_tool not available", elapsed_ms
        
        result = self.tool_registry["cv_assess_rag"](**args)
        
        # Store CV results
        if "cv" in result:
            state.evidence["cv"] = result["cv"]
        
        # Store support documents
        if "support" in result:
            state.evidence["support"] = result["support"]
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"[Executor] cv_assess_rag: analysis complete")
        
        return True, None, elapsed_ms


# ========== CONVENIENCE FUNCTION ==========
def execute_plan(
    tool_chain: List[Dict[str, Any]],
    slots: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a tool chain (backwards compatible interface)
    
    Args:
        tool_chain: List of steps with {"tool": str, "args": dict}
        slots: User intent slots
    
    Returns:
        Execution state as dictionary
    """
    executor = ToolExecutor()
    state = executor.execute(tool_chain, slots)
    return state.to_dict()