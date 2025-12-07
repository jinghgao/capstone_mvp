from Data_layer import DataLayer
import sqlite3
import time
from typing import Dict, Any
# ==========================================
# IMPORTS
# ==========================================
import ragas.metrics as ragas_metrics
from ragas import evaluate as ragas_evaluate
from ragas.dataset_schema import EvaluationDataset
import numpy as np


# ==========================================
# MOCK SYSTEM — Replace with your functions
# ==========================================

def rag_retrieve(query, top_k: int = 3, filters: Dict[str, Any] | None = None):
    """Run KB retrieval via the ToolExecutor._exec_kb_retrieve handler.

    This uses the same executor path as the agent runtime so results
    are stored in an ExecutionState and logged similarly.
    Returns the list of hits (each a dict with 'text', 'source', etc.).
    """
    from executor import ToolExecutor, ExecutionState

    execr = ToolExecutor()
    state = ExecutionState(slots={})
    args = {"query": query, "top_k": top_k, "filters": filters}
    start = time.time()
    success, error, elapsed = execr._exec_kb_retrieve(args, state, start)
    if not success:
        print(f"[rag_retrieve] kb_retrieve failed: {error}")
        return []
    return state.evidence.get("kb_hits", [])


def llm_reason(context, question, sql: str = ""):
    """Use composer._summarize_rag_context_dimension_comparison to produce an LLM-style answer.

    context: list[dict] or list[str] (RAG hits or plain texts)
    question: user query
    sql: optional SQL string to include in the prompt
    """
    try:
        from composer import _summarize_rag_context_dimension_comparison
    except Exception:
        # Fallback to simple behavior if composer not available
        if isinstance(context, list) and context and isinstance(context[0], dict):
            texts = [h.get("text", "") for h in context]
        else:
            texts = list(context or [])
        return " ".join(t[:200] for t in texts)[:500]

    # Normalize context into list[dict]
    rag_snippets = []
    if isinstance(context, list):
        if context and isinstance(context[0], dict):
            rag_snippets = context
        else:
            rag_snippets = [{"text": str(c)} for c in context]
    else:
        rag_snippets = [{"text": str(context)}]
    # print("RAG snippets:", rag_snippets)
    # Call the composer summarizer (it handles LLM availability and fallbacks)
    try:
        return _summarize_rag_context_dimension_comparison(rag_snippets, question, sql_result_summary=sql, sql="", field_type="diamond")
    except Exception as e:
        return f"[llm_reason error] {e}"


# ==========================================
# GROUND TRUTH DATASET
# ==========================================
def generate_query(field_data, standard):
    dataset = []
    for field in field_data: 
        field_name = field.get("Name of Field", "Unknown Field")
        for std in standard:
            retrieval_ground_truth = std
            if field.get("Diamonds: Dimension Home to Pitchers Plate - m") is None or field.get("Diamonds: Home to First Base Path - m") is None:
                continue
            answer_ground_truth = {"Home to Pitchers Plate" : True if std.get("Dimension Home to Pitchers Plate - should greater than") <= field.get("Diamonds: Dimension Home to Pitchers Plate - m") <= std.get("Dimension Home to Pitchers Plate - should less than") else False,
                                   "Home to First Base Path" : True if std.get("Home to First Base Path - should greater than") <= field.get("Diamonds: Home to First Base Path - m") <= std.get("Home to First Base Path - should less than") else False}
            if answer_ground_truth["Home to Pitchers Plate"] is False:
                answer_ground_truth["Pitchers Plate Difference"] = field.get("Diamonds: Dimension Home to Pitchers Plate - m") - std.get("Dimension Home to Pitchers Plate - should greater than") if field.get("Diamonds: Dimension Home to Pitchers Plate - m") < std.get("Dimension Home to Pitchers Plate - should greater than") else field.get("Diamonds: Dimension Home to Pitchers Plate - m") - std.get("Dimension Home to Pitchers Plate - should less than")
            if answer_ground_truth["Home to First Base Path"] is False:
                answer_ground_truth["First Base Path Difference"] = field.get("Diamonds: Home to First Base Path - m") - std.get("Home to First Base Path - should greater than") if field.get("Diamonds: Home to First Base Path - m") < std.get("Home to First Base Path - should greater than") else field.get("Diamonds: Home to First Base Path - m") - std.get("Home to First Base Path - should less than")
            
            dataset.append({
                "query": f"Compare {field_name} dimensions to {std.get('sport')} standards.",
                "retrieval_ground_truth": retrieval_ground_truth,
                "answer_ground_truth": answer_ground_truth,
                "retrieved_field_data": field
            })
    return dataset

# ==========================================
# SECTION 1 — RAG RETRIEVAL EVALUATION
# ==========================================

def evaluate_retrieval(dataset, llm=None, metrics=None):
    # Build ragas EvaluationDataset (SingleTurnSample-like entries)
    data_list = []
    for item in dataset[:10]:
        retrieved_docs = rag_retrieve(item["query"])
        # normalize to list of strings for ragas / LLM inputs
        retrieved_texts = [h.get("text", "") for h in retrieved_docs]
        # include a placeholder model response so default ragas metrics can run
        model_resp = llm_reason(retrieved_texts, item["query"])
        print("Model response:", model_resp)

        # Ensure reference_contexts is a list of strings (ragas expects a list)
        ref_ctx = item.get("retrieval_ground_truth", [])
        if isinstance(ref_ctx, dict):
            # single dict -> stringify
            try:
                import json as _json
                ref_list = [_json.dumps(ref_ctx, ensure_ascii=False)]
            except Exception:
                ref_list = [str(ref_ctx)]
        elif isinstance(ref_ctx, list):
            # convert any dict elements to strings
            ref_list = []
            for r in ref_ctx:
                if isinstance(r, dict):
                    try:
                        import json as _json
                        ref_list.append(_json.dumps(r, ensure_ascii=False))
                    except Exception:
                        ref_list.append(str(r))
                else:
                    ref_list.append(str(r))
        else:
            ref_list = [str(ref_ctx)]

        data_list.append({
            "user_input": item["query"],
            "retrieved_contexts": retrieved_texts,
            "reference_contexts": ref_list,
            "reference": str(item.get("answer_ground_truth", "")),
            "response": model_resp,
        })
    ragas_dataset = EvaluationDataset.from_list(data_list)

    # If caller didn't supply metrics, prefer retrieval-only metrics to avoid requiring an LLM.
    if metrics is None:
        try:
            metrics = []
            if hasattr(ragas_metrics, "precision_at_k"):
                metrics.append(ragas_metrics.precision_at_k(1))
            if hasattr(ragas_metrics, "recall_at_k"):
                metrics.append(ragas_metrics.recall_at_k(3))
            if hasattr(ragas_metrics, "mean_reciprocal_rank"):
                metrics.append(ragas_metrics.mean_reciprocal_rank())
            # optionally add nDCG if available
            if hasattr(ragas_metrics, "ndcg"):
                try:
                    metrics.append(ragas_metrics.ndcg(k=3))
                except TypeError:
                    # some versions expect no args
                    metrics.append(ragas_metrics.ndcg())
        except Exception:
            metrics = None

    # Call ragas.evaluate; pass llm if provided (enables LLM-based metrics),
    # otherwise evaluation will run with the selected metrics only.
    try:
        if metrics is not None:
            ragas_results = ragas_evaluate(ragas_dataset, metrics=metrics, llm=llm)
        else:
            ragas_results = ragas_evaluate(ragas_dataset, llm=llm)
        return ragas_results
    except Exception as e:
        # Some ragas metric runners raise inside worker threads; retry without raising
        print(f"[WARN] ragas.evaluate raised an exception: {e}")
        # Fallback: compute basic retrieval metrics locally (precision@1, recall@3, MRR, nDCG@3)
        print("[INFO] Falling back to local retrieval metric computation (precision@1, recall@3, mrr, ndcg@3)")

        def _is_relevant(hit_text: str, refs: list[str]) -> bool:
            if not hit_text:
                return False
            ht = hit_text.lower()
            for r in refs:
                if not r:
                    continue
                try:
                    if isinstance(r, str) and r.lower() in ht:
                        return True
                except Exception:
                    if str(r).lower() in ht:
                        return True
            return False

        top_k = 3
        precisions_at_1 = []
        recalls_at_3 = []
        mrrs = []
        ndcgs = []

        for sample in data_list:
            retrieved = sample.get("retrieved_contexts", [])[:top_k]
            refs = sample.get("reference_contexts", [])

            # precision@1
            p1 = 0.0
            if retrieved:
                p1 = 1.0 if _is_relevant(retrieved[0], refs) else 0.0
            precisions_at_1.append(p1)

            # recall@3
            recall = 0.0
            for r in retrieved:
                if _is_relevant(r, refs):
                    recall = 1.0
                    break
            recalls_at_3.append(recall)

            # MRR
            rr = 0.0
            for idx, r in enumerate(retrieved):
                if _is_relevant(r, refs):
                    rr = 1.0 / (idx + 1)
                    break
            mrrs.append(rr)

            # nDCG@3 (binary relevance)
            import math
            dcg = 0.0
            for idx, r in enumerate(retrieved, start=1):
                rel = 1.0 if _is_relevant(r, refs) else 0.0
                if rel > 0:
                    dcg += rel / math.log2(idx + 1)
            # ideal DCG for single relevant doc is 1 / log2(1+1) = 1
            idcg = 1.0 if any(_is_relevant(r, refs) for r in retrieved) else 0.0
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        results = {
            "precision_at_1": float(np.mean(precisions_at_1)) if precisions_at_1 else 0.0,
            "recall_at_3": float(np.mean(recalls_at_3)) if recalls_at_3 else 0.0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "ndcg_at_3": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "note": "computed locally due to ragas runner exception"
        }
        return results


# ==========================================
# SECTION 2 — LLM REASONING / CALCULATION EVALUATION
# ==========================================

class NumericalCorrectness:
    """Custom numeric metric for comparing model JSON-based calculations."""

    def measure(self, expected, actual):
        score = 0
        total = 0

        for key in expected:
            total += 1
            if key not in actual:
                continue

            if isinstance(expected[key], bool):
                score += int(expected[key] == actual[key])
            else:
                diff = abs(expected[key] - actual[key])
                score += (1 - min(diff / 10, 1))  # allow small tolerance

        return score / total


def evaluate_reasoning(dataset):
    # Lightweight heuristic evaluator: parse LLM output for expected numeric/boolean values.
    import re
    results = []
    for item in dataset[:5]:
        retrieved_docs = rag_retrieve(item["query"]) 
        retrieved_texts = [h.get("text", "") for h in retrieved_docs]
        print("Retrieved texts:", retrieved_texts)
        # print("Reference: ", item.get("retrieval_ground_truth", {}))
        # print("query: ", item["query"])
        # print("retrieved_dimention: ", item.get("retrieved_field_data", ""))
        # llm_output = llm_reason(item.get("retrieval_ground_truth", []), item["query"], sql=item.get("retrieved_field_data", ""))
        # print("LLM output:", llm_output)
        llm_output = llm_reason(retrieved_texts, item["query"], sql=item.get("retrieved_field_data", ""))
        expected = item.get("answer_ground_truth", {}) or {}
        score = 0.0
        checks = 0

        # Normalize llm_output into a dict if possible (dict, JSON, or Python literal)
        parsed = None
        if isinstance(llm_output, dict):
            parsed = llm_output
        else:
            try:
                import json as _json
                parsed = _json.loads(llm_output)
            except Exception:
                try:
                    import ast as _ast
                    parsed = _ast.literal_eval(llm_output)
                except Exception:
                    parsed = None

        # Expected keys from composer
        bool_keys = ["Home to Pitchers Plate", "Home to First Base Path"]
        diff_keys = ["Pitchers Plate Difference", "First Base Path Difference"]

        # Boolean comparisons
        for k in bool_keys:
            if k in expected:
                checks += 1
                exp_val = expected.get(k)
                actual = None
                if isinstance(parsed, dict):
                    actual = parsed.get(k)
                else:
                    low = (str(llm_output) or "").lower()
                    if k.lower() in low:
                        if "true" in low or "yes" in low:
                            actual = True
                        elif "false" in low or "no" in low:
                            actual = False

                if actual is True or actual is False:
                    if actual == exp_val:
                        score += 1.0

        # Numeric difference comparisons
        for k in diff_keys:
            if k in expected:
                checks += 1
                exp_val = expected.get(k)
                actual_val = None
                if isinstance(parsed, dict):
                    actual_val = parsed.get(k)
                else:
                    m = re.search(re.escape(k) + r"[^0-9\-\.]*([-+]?[0-9]*\.?[0-9]+)", str(llm_output))
                    if m:
                        try:
                            actual_val = float(m.group(1))
                        except Exception:
                            actual_val = None

                if exp_val is None:
                    if actual_val is None:
                        score += 1.0
                else:
                    try:
                        if actual_val is not None and abs(float(actual_val) - float(exp_val)) <= 1.0:
                            score += 1.0
                    except Exception:
                        pass

        results.append(score / max(checks, 1))

    return results


# ==========================================
# SECTION 3 — END-TO-END PIPELINE EVALUATION
# ==========================================

def evaluate_pipeline(dataset):
    # Simple end-to-end heuristic evaluation: run retrieval + LLM and score with same
    # lightweight checks used in evaluate_reasoning.
    import re
    results = []
    for item in dataset[10:20]:
        retrieved_docs = rag_retrieve(item["query"])
        retrieved_texts = [h.get("text", "") for h in retrieved_docs]
        llm_output = llm_reason(retrieved_texts, item["query"], sql=item.get("retrieved_field_data", ""))

        expected = item.get("answer_ground_truth", {}) or {}
        score = 0.0
        checks = 0

        # Try to parse LLM output into a dict (JSON or Python literal) if possible
        parsed = None
        if isinstance(llm_output, dict):
            parsed = llm_output
        else:
            try:
                import json as _json
                parsed = _json.loads(llm_output)
                print(f"[DEBUG] Parsed JSON:\n{parsed}\n")
            except Exception:
                try:
                    import ast as _ast
                    parsed = _ast.literal_eval(llm_output)
                except Exception:
                    parsed = None

        bool_keys = ["Home to Pitchers Plate", "Home to First Base Path"]
        diff_keys = ["Pitchers Plate Difference", "First Base Path Difference"]

        for k in bool_keys:
            if k in expected:
                checks += 1
                exp_val = expected.get(k)
                actual = None
                if isinstance(parsed, dict):
                    actual = parsed.get(k)
                else:
                    low = (str(llm_output) or "").lower()
                    if k.lower() in low:
                        if "true" in low or "yes" in low:
                            actual = True
                        elif "false" in low or "no" in low:
                            actual = False
                if actual is True or actual is False:
                    if actual == exp_val:
                        score += 1.0

        for k in diff_keys:
            if k in expected:
                checks += 1
                exp_val = expected.get(k)
                actual_val = None
                if isinstance(parsed, dict):
                    actual_val = parsed.get(k)
                else:
                    m = re.search(re.escape(k) + r"[^0-9\-\.]*([-+]?[0-9]*\.?[0-9]+)", str(llm_output))
                    if m:
                        try:
                            actual_val = float(m.group(1))
                        except Exception:
                            actual_val = None
                if exp_val is None:
                    if actual_val is None:
                        score += 1.0
                else:
                    try:
                        if actual_val is not None and abs(float(actual_val) - float(exp_val)) <= 1.0:
                            score += 1.0
                    except Exception:
                        pass

        results.append({"query": item["query"], 
                        "retrieved_field_data": item.get("retrieved_field_data"),
                        "answer_ground_truth": expected, 
                        "output": llm_output,
                        "retrieval_ground_truth": item.get("retrieval_ground_truth"), 
                        "retrieved_text": retrieved_texts, 
                        "score": score,
                        "checks": checks,
                        "score/checks": score / max(checks, 1)})
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_excel("experiment/pipeline_evaluation_results_llama3:8b.xlsx", index=False)
    return results


# ==========================================
# RUN ALL EVALUATIONS
# ==========================================





def _tpl_get_diamond_dimensions(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns diamond field dimensions from the diamond_field_size_data table.
    """
    sql = f"""
    SELECT "Name of Field", "Diamonds: Dimension Home to Pitchers Plate - m", "Diamonds: Home to First Base Path - m"
    FROM diamond_field_size_data
    LIMIT 10;
    """
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}

def _tpl_get_diamond_fields_names(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns diamond field dimensions from the diamond_field_size_data table.
    """
    sql = """
    SELECT "Name of Field"
    FROM diamond_field_size_data;
    """
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}
def _tpl_get_rectangular_fields_names(con: sqlite3.Connection, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns diamond field dimensions from the diamond_field_size_data table.
    """
    sql = """
    SELECT "Name of Field"
    FROM rectangular_field_size_data;
    """
    t0 = time.time()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    elapsed = int((time.time() - t0) * 1000)
    return {"rows": rows, "rowcount": len(rows), "elapsed_ms": elapsed}
dl = DataLayer()
dl.initialize_database()
field_data = _tpl_get_diamond_dimensions(dl.get_connection(), None)["rows"]
field_names = _tpl_get_diamond_fields_names(dl.get_connection(), None)["rows"]
names = [row["Name of Field"] for row in field_names]
rect_fields_name = _tpl_get_rectangular_fields_names(dl.get_connection(), None)["rows"]
rect_names = [row["Name of Field"] for row in rect_fields_name]
standard = [{
    "sport": "Softball Men U15",
    "Dimension Home to Pitchers Plate - should greater than": 12.5,
    "Dimension Home to Pitchers Plate - should less than": 13.1,
    "Home to First Base Path - should greater than": 17.988,
    "Home to First Base Path - should less than": 18.588,
}, {
    "sport": "Softball Men U13",
    "Dimension Home to Pitchers Plate - should greater than": 11.2,
    "Dimension Home to Pitchers Plate - should less than": 11.8,
    "Home to First Base Path - should greater than": 16.5,
    "Home to First Base Path - should less than": 17.04,
}, {
    "sport": "Softball Men U11",
    "Dimension Home to Pitchers Plate - should greater than": 10.37,
    "Dimension Home to Pitchers Plate - should less than": 10.97,
    "Home to First Base Path - should greater than": 13.5,
    "Home to First Base Path - should less than": 14.02,
}, {
    "sport": "Softball Men U9, U7",
    "Dimension Home to Pitchers Plate - should greater than": 8.844,
    "Dimension Home to Pitchers Plate - should less than": 9.44,
    "Home to First Base Path - should greater than": 13.5,
    "Home to First Base Path - should less than": 14.02,
}]




if __name__ == "__main__":
    dl = DataLayer().initialize_database()
    evaluation_dataset = generate_query(_tpl_get_diamond_dimensions(dl.get_connection(), None)["rows"], standard)

    print("\n==== 🟦 RAG RETRIEVAL EVALUATION ====")
    # retrieval_results = evaluate_retrieval(evaluation_dataset)
    # print(retrieval_results)

    print("\n==== 🟩 LLM REASONING EVALUATION ====")
    # reasoning_results = evaluate_reasoning(evaluation_dataset)
    # print("average reasoning score:", sum(reasoning_results) / len(reasoning_results) if reasoning_results else 0.0)
    # print("Reasoning scores:", reasoning_results)

    print("\n==== 🟧 END-TO-END PIPELINE EVALUATION ====")
    pipeline_results = evaluate_pipeline(evaluation_dataset)
    print(pipeline_results)