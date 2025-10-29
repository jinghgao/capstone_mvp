# composer.py - Response Composition Layer (Adapted for Refactored Architecture)
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

# ========== LLM Integration (Ollama Only) ==========
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[WARN] OpenAI library not available. Install: pip install openai")
    print("[INFO] Note: We use OpenAI library to connect to Ollama (local LLM)")

# Configuration - Local Ollama Only
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"


def _summarize_rag_context(
    rag_snippets: List[Dict[str, Any]],
    query: str,
    sql_result_summary: str = ""
) -> str:
    """
    Use local Ollama LLM to summarize RAG document snippets into coherent context.
    Falls back to simple formatting if LLM is unavailable or fails.
    """
    if not rag_snippets:
        return ""

    if not LLM_AVAILABLE:
        return _format_rag_snippets_simple(rag_snippets)

    try:
        context_text = "\n\n".join([
            f"Source {i+1} (page {snippet.get('page', '?')}): {snippet.get('text', '')[:500]}"
            for i, snippet in enumerate(rag_snippets[:3])
        ])

        if sql_result_summary:
            prompt = f"""You are an assistant helping interpret park maintenance data.

User Question: {query}

SQL Query Result: {sql_result_summary}

Reference Documents:
{context_text}

Task: Based on the reference documents, provide 2-3 sentences of relevant context that helps interpret the SQL results above. Focus on:
- Relevant standards, procedures, or guidelines
- Cost factors or typical ranges mentioned
- Any important notes about the data

Keep it concise and directly relevant to the user's question. Use markdown formatting."""
        else:
            prompt = f"""You are an assistant helping answer questions about park maintenance procedures.

User Question: {query}

Reference Documents:
{context_text}

Task: Summarize the key information from the reference documents that answers the user's question. Provide:
- 2-3 key points or steps
- Relevant standards or guidelines
- Important safety notes if applicable

Use markdown formatting with bullet points."""

        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes technical documentation clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        print(f"[WARN] Ollama LLM summarization failed: {e}")
        print(f"[INFO] Make sure Ollama is running and model is available (e.g., `ollama list`).")
        return _format_rag_snippets_simple(rag_snippets)


def _format_rag_snippets_simple(snippets: List[Dict[str, Any]]) -> str:
    """Simple RAG snippet formatting (fallback when LLM is unavailable)."""
    if not snippets:
        return ""

    output = "### Reference Context\n\n"
    for i, snippet in enumerate(snippets[:3], 1):
        text = snippet.get("text", "")
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:200] + "..." if len(text) > 200 else text
        page = snippet.get("page", "?")
        output += f"**Source {i}** (page {page}):\n{text}\n\n"
    return output


def _snip(txt: str, n: int = 150) -> str:
    """Truncate text and clean whitespace."""
    s = re.sub(r"\s+", " ", (txt or "")).strip()
    return (s[:n] + "...") if len(s) > n else s


def compose_answer(
    nlu: Dict[str, Any],
    state: Dict[str, Any],
    plan_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compose final user-facing answer from execution state.

    Returns:
        {
            "answer_md": str,
            "tables": list,
            "charts": list,
            "citations": list,
            "logs": list
        }
    """
    # --- Early returns for unsupported or clarification flows ---
    status = state.get("status") or (plan_metadata or {}).get("status")
    if status == "UNSUPPORTED":
        msg = state.get("message") or "This question is not supported yet"
        return {
            "answer_md": f"**Not Supported**\n\n{msg}",
            "tables": [],
            "charts": [],
            "map_layer": None,
            "citations": [],
            "logs": state.get("logs", [])
        }

    clarifications = state.get("clarifications") or []
    if status == "NEEDS_CLARIFICATION" and clarifications:
        bullets = "\n".join([f"- {q}" for q in clarifications])
        return {
            "answer_md": f"**I need a bit more info to proceed:**\n\n{bullets}",
            "tables": [],
            "charts": [],
            "map_layer": None,
            "citations": [],
            "logs": state.get("logs", [])
        }
    # --- End early returns ---

    intent = nlu.get("intent", "")
    ev = state.get("evidence", {})
    tables: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    charts: List[Dict[str, Any]] = []
    answer_md = ""

    # Get user's original query
    user_query = (
        nlu.get("raw_query", "") or
        nlu.get("slots", {}).get("text", "") or
        state.get("slots", {}).get("text", "")
    )

    # Template hint from planner
    template_hint = (plan_metadata or {}).get("template")
    if not template_hint:
        template_hint = _extract_template_from_logs(state.get("logs", []))

    # NLU slots (for summaries / titles)
    slots = nlu.get("slots", {}) or state.get("slots", {})
    explanation_requested = bool(slots.get("explanation_requested"))

    # Optional header (workflow + template)
    header = []
    if intent:
        header.append(f"**Workflow:** `{intent}`")
    if template_hint:
        header.append(f"**Template:** `{template_hint}`")
    if header:
        answer_md = "  •  ".join(header) + "\n\n"

    # ========== RAG content handling ==========
    if intent in ("RAG", "RAG+SQL_tool", "RAG+CV_tool"):
        sop = ev.get("sop", {})
        kb_hits = ev.get("kb_hits", [])
        has_sop_content = any(sop.get(k) for k in ["steps", "materials", "tools", "safety"])

        # Heuristic domain hint
        is_mowing_query = False
        if kb_hits:
            combined_text = " ".join([h.get("text", "")[:200] for h in kb_hits[:2]]).lower()
            is_mowing_query = any(k in combined_text for k in ["mowing", "mow", "equipment", "ppe", "inspection"])

        if has_sop_content and is_mowing_query:
            # SOP presentation
            answer_md += "**Mowing SOP (Standard Operating Procedures)**\n\n"
            if sop.get("steps"):
                answer_md += "### Steps\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(sop["steps"])])
            if sop.get("materials"):
                answer_md += "\n\n### Materials\n- " + "\n- ".join(sop["materials"])
            if sop.get("tools"):
                answer_md += "\n\n### Tools\n- " + "\n- ".join(sop["tools"])
            if sop.get("safety"):
                answer_md += "\n\n### Safety\n- " + "\n- ".join(sop["safety"])

            for h in kb_hits[:3]:
                citations.append({"title": "Maintenance Standard/Manual", "source": h.get("source", "")})

        elif kb_hits and intent == "RAG":
            # Generic RAG (no SQL)
            answer_md += "### Reference Summary\n\n"
            if LLM_AVAILABLE:
                try:
                    summary = _summarize_rag_context(
                        rag_snippets=kb_hits,
                        query=user_query or "Maintenance standards query",
                        sql_result_summary=""
                    )
                    answer_md += summary
                except Exception as e:
                    print(f"[WARN] LLM summary failed: {e}")
                    answer_md += _format_rag_snippets_simple(kb_hits)
            else:
                answer_md += _format_rag_snippets_simple(kb_hits)

            for h in kb_hits[:3]:
                citations.append({"title": "Reference", "source": h.get("source", "")})

    # ========== SQL content handling ==========
    if intent in ("SQL_tool", "RAG+SQL_tool"):
        sql = ev.get("sql", {})
        rows = sql.get("rows", [])

        # Chart
        chart_config = _detect_chart_type(rows, template_hint)
        if chart_config:
            charts.append(chart_config)

        # Table
        if rows:
            tables.append({
                "name": _get_table_name(template_hint, slots),
                "columns": list(rows[0].keys()),
                "rows": rows
            })

        # SQL summary
        sql_summary = _generate_sql_summary(rows, template_hint, slots)
        answer_md += ("" if answer_md.endswith("\n\n") or not answer_md else "\n\n") + sql_summary
        answer_md += f"\n\n**Query Performance**: {sql.get('rowcount', 0)} rows in {sql.get('elapsed_ms', 0)} ms"

        # RAG context only for hybrid queries (and only if explanation was requested)
        if intent == "RAG+SQL_tool" and explanation_requested:
            rag_hits = ev.get("kb_hits", [])
            if rag_hits:
                answer_md += "\n\n---\n\n"
                rag_context = _summarize_rag_context(
                    rag_snippets=rag_hits,
                    query=user_query or "",
                    sql_result_summary=sql_summary
                )
                answer_md += rag_context
                for h in rag_hits[:3]:
                    citations.append({"title": "Reference Document", "source": h.get("source", "")})

    # ========== CV content handling ==========
    if intent in ("CV_tool", "RAG+CV_tool"):
        cv = ev.get("cv", {})
        is_mock = any("VLM not configured" in str(label) for label in cv.get("labels", [])) if isinstance(cv.get("labels", []), list) else False

        if is_mock:
            answer_md += (
                ("\n\n" if answer_md else "") +
                "**Image Analysis (Not Configured)**\n\n"
                "To enable AI-powered image analysis:\n"
                "1. Get an API key from your provider (e.g., OpenRouter)\n"
                "2. Set environment variable: `export OPENROUTER_API_KEY='your-key'`\n"
                "3. Restart backend\n\n"
                "Supported analysis:\n"
                "- Field condition assessment\n"
                "- Turf health evaluation\n"
                "- Maintenance recommendations\n"
                "- Safety hazard detection"
            )
        else:
            answer_md += (
                ("\n\n" if answer_md else "") +
                "**Image Assessment**\n\n"
                f"Condition: **{cv.get('condition','unknown')}** (score {cv.get('score',0):.2f})\n\n"
                f"Issues: {', '.join(cv.get('labels', [])) if isinstance(cv.get('labels', []), list) else cv.get('labels', '')}\n\n"
                f"Recommendations: {'; '.join(cv.get('explanations', [])) if isinstance(cv.get('explanations', []), list) else cv.get('explanations', '')}"
            )
            if cv.get("low_confidence"):
                answer_md = "> ⚠️ Low confidence - consider uploading a clearer image.\n\n" + answer_md

        # Add RAG context for RAG+CV (when VLM is active)
        if intent == "RAG+CV_tool" and not is_mock:
            rag_hits = ev.get("kb_hits", [])
            if rag_hits:
                answer_md += "\n\n---\n\n"
                rag_context = _summarize_rag_context(
                    rag_snippets=rag_hits,
                    query=user_query or "",
                    sql_result_summary=""
                )
                answer_md += rag_context

        for h in ev.get("support", [])[:2]:
            citations.append({"title": "Reference Standards", "source": h.get("source", "")})

    # ========== Fallback ==========
    if not answer_md:
        answer_md = "I couldn't generate a response for this query."

    return {
        "answer_md": answer_md,
        "tables": tables,
        "charts": charts,
        "map_layer": None,
        "citations": citations,
        "logs": state.get("logs", [])
    }


# ========== Helper Functions ==========

def _extract_template_from_logs(logs: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract SQL template from execution logs (best-effort).
    """
    # Currently we do not log template in args; prefer plan_metadata.
    # Keep this stub for future enrichment.
    return None


def _get_table_name(template_hint: Optional[str], slots: Dict[str, Any]) -> str:
    """Generate table name based on template."""
    if not template_hint:
        return "Query Result"

    if template_hint == "mowing.labor_cost_month_top1":
        month = slots.get("month", "")
        year = slots.get("year", "")
        return f"Top Park by Mowing Cost ({month}/{year})"
    elif template_hint == "mowing.cost_trend":
        return "Mowing Cost Trend"
    elif template_hint == "mowing.cost_by_park_month":
        return "Cost Comparison by Park"
    elif template_hint == "mowing.last_mowing_date":
        return "Last Mowing Dates"
    elif template_hint == "mowing.cost_breakdown":
        return "Detailed Cost Breakdown"

    return "Query Result"


def _generate_sql_summary(rows: List[Dict], template_hint: Optional[str], slots: Dict[str, Any]) -> str:
    """Generate natural language summary of SQL results."""
    if not rows:
        return "No results found."

    if template_hint == "mowing.labor_cost_month_top1":
        park = rows[0].get("park", "Unknown")
        cost = rows[0].get("total_cost", 0)
        month = slots.get("month", "")
        year = slots.get("year", "")
        return f"### 🏆 Results\n\n**{park}** had the highest mowing cost of **${cost:,.2f}** in {month}/{year}."

    elif template_hint == "mowing.cost_trend":
        return f"### 📈 Trend Analysis\n\nCost trend data across **{len(rows)} time periods**."

    elif template_hint == "mowing.cost_by_park_month":
        total = sum(row.get("total_cost", 0) for row in rows)
        return f"### 📊 Cost Comparison\n\n**{len(rows)} parks** with combined costs of **${total:,.2f}**."

    elif template_hint == "mowing.last_mowing_date":
        return f"### 📅 Last Mowing Activity\n\nShowing data for **{len(rows)} park(s)**."

    elif template_hint == "mowing.cost_breakdown":
        return f"### 💰 Detailed Breakdown\n\n**{len(rows)} cost entries** by activity type."

    return f"### Results\n\nFound **{len(rows)} records**."


def _detect_chart_type(rows: List[Dict], template_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Detect appropriate chart type based on data structure."""
    if not rows:
        return None

    columns = list(rows[0].keys())

    if template_hint == "mowing.cost_trend":
        if "month" in columns and "monthly_cost" in columns:
            parks = sorted(list(set(row.get("park") for row in rows if row.get("park"))))
            if len(parks) > 10:
                park_totals = {}
                for park in parks:
                    park_totals[park] = sum(
                        row.get("monthly_cost", 0) for row in rows
                        if row.get("park") == park
                    )
                top_parks = sorted(park_totals.items(), key=lambda x: x[1], reverse=True)[:10]
                parks = [p[0] for p in top_parks]

            return {
                "type": "line",
                "title": "Mowing Cost Trend",
                "x_axis": {"field": "month", "label": "Month", "type": "category"},
                "y_axis": {"field": "monthly_cost", "label": "Cost ($)", "type": "value"},
                "series": [
                    {
                        "name": park,
                        "data": [
                            {"x": row["month"], "y": row["monthly_cost"]}
                            for row in rows if row.get("park") == park
                        ]
                    }
                    for park in parks
                ],
                "legend": True,
                "grid": True
            }

    elif template_hint in ["mowing.cost_by_park_month", "mowing.labor_cost_month_top1"]:
        if "park" in columns and "total_cost" in columns:
            return {
                "type": "bar",
                "title": "Mowing Cost by Park",
                "x_axis": {"field": "park", "label": "Park", "type": "category"},
                "y_axis": {"field": "total_cost", "label": "Total Cost ($)", "type": "value"},
                "series": [
                    {
                        "name": "Total Cost",
                        "data": [{"x": row["park"], "y": row["total_cost"]} for row in rows]
                    }
                ],
                "legend": False,
                "grid": True
            }

    elif template_hint == "mowing.last_mowing_date":
        if "park" in columns or "PARK" in columns:
            def get_field(row, *field_names):
                for field in field_names:
                    if field in row:
                        return row[field]
                    if field.upper() in row:
                        return row[field.upper()]
                    if field.lower() in row:
                        return row[field.lower()]
                return None

            return {
                "type": "timeline",
                "title": "Last Mowing Date by Park",
                "data": [
                    {
                        "park": get_field(row, "park", "PARK"),
                        "date": get_field(row, "last_mowing_date", "LAST_MOWING_DATE"),
                        "sessions": get_field(row, "total_sessions", "TOTAL_SESSIONS", "total_mowing_sessions", "TOTAL_MOWING_SESSIONS"),
                        "cost": get_field(row, "total_cost", "TOTAL_COST")
                    }
                    for row in rows
                ],
                "sort_by": "date",
                "sort_order": "desc"
            }

    return None


def _generate_chart_description(chart_config: Dict[str, Any], rows: List[Dict]) -> str:
    """Generate chart description text."""
    if not chart_config or not rows:
        return ""

    chart_type = chart_config.get("type")

    if chart_type == "line":
        parks = list(set(row.get("park") for row in rows if row.get("park")))
        months = sorted(set(row.get("month") for row in rows if row.get("month")))
        return f"Line chart comparing {len(parks)} park(s) from month {min(months)} to {max(months)}"

    elif chart_type == "bar":
        return f"Bar chart comparing {len(rows)} park(s)"

    elif chart_type == "timeline":
        return f"Timeline of last mowing dates for {len(rows)} park(s)"

    return ""