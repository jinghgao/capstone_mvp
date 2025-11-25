# planner.py - Execution Planning Layer
# Responsibility: Convert NLU results into executable tool plans
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
from rapidfuzz import process
from nlu import NLUResult


# ========== EXECUTION PLAN DATACLASS ==========
class ExecutionPlan:
    """Structured execution plan"""
    def __init__(
        self,
        tool_chain: List[Dict[str, Any]],
        clarifications: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.tool_chain = tool_chain
        self.clarifications = clarifications
        self.metadata = metadata or {}

    def is_ready(self) -> bool:
        """Check if plan is ready to execute (no clarifications needed)"""
        return len(self.clarifications) == 0 and self.metadata.get("status") != "UNSUPPORTED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tool_chain": self.tool_chain,
            "clarifications": self.clarifications,
            "metadata": self.metadata,
            "ready": self.is_ready()
        }


# ========== EXECUTION PLANNER ==========
class ExecutionPlanner:
    """
    Converts NLU results into concrete execution plans.
    - No intent rewriting here.
    - If a SQL intent cannot be mapped to a supported template -> mark UNSUPPORTED.
    - If required params are missing for a supported template -> ask clarifications.
    """

    def __init__(self):
        # SQL template registry
        self.sql_templates = {
            "mowing.labor_cost_month_top1": {
                "builder": self._build_top_cost_params,
                "required": ["month", "year"],
                "optional": ["park_name"]
            },
            "mowing.cost_trend": {
                "builder": self._build_trend_params,
                "required": ["start_month", "end_month", "year"],
                "optional": ["park_name"]
            },
            "mowing.last_mowing_date": {
                "builder": self._build_last_date_params,
                "required": [],
                "optional": ["park_name"]
            },
            "mowing.cost_by_park_month": {
                "builder": self._build_cost_by_park_params,
                "required": ["month", "year"],
                "optional": ["park_name"]
            },
            "mowing.cost_breakdown": {
                "builder": self._build_breakdown_params,
                "required": ["month", "year"],
                "optional": ["park_name"]
            },
            "mowing.cost_by_park_least_per_sqft": {
                "builder": self._build_cost_least_per_sqft_params,
                "required": ["month1", "month2", "year1", "year2"],
                "optional": []
            },
            "field_dimension.rectangular": {
                "builder": self._build_field_dimension_params_rectangular,  # Placeholder
                "required": ["field_name"],
                "optional": []
            },
            "field_dimension.diamond": {
                "builder": self._build_field_dimension_params_diamond,  # Placeholder
                "required": ["field_name"],
                "optional": []
            },
            "activity.cost_by_location_range": {
                "builder": self._build_activity_cost_params,
                "required": ["park_name", "month1", "month2", "year1", "year2"],
                "optional": ["activity_name"]
            },
            "activity.last_activity_date": {
                "builder": self._build_last_activity_date_params,
                "required": ["park_name"],
                "optional": ["activity_name"]
            },
            "activity.maintenance_due_window": {
                "builder": self._build_activity_due_params,
                "required": ["activity_name", "weeks_ahead"],
                "optional": ["park_name"]
            }
        }

    # ========== MAIN PLANNING ENTRY ==========
    def plan(self, nlu_result: NLUResult) -> ExecutionPlan:
        """
        Generate execution plan from NLU result (no intent rewriting).

        Returns:
            ExecutionPlan with tool chain and clarifications
        """
        intent = nlu_result.intent

        if intent == "RAG":
            return self._plan_rag(nlu_result)

        if intent == "SQL_tool":
            return self._plan_sql(nlu_result)

        if intent == "RAG+SQL_tool":
            return self._plan_rag_sql(nlu_result)

        if intent == "CV_tool":
            return self._plan_cv(nlu_result)

        if intent == "RAG+CV_tool":
            return self._plan_rag_cv(nlu_result)

        return ExecutionPlan(
            tool_chain=[],
            clarifications=[],
            metadata={"status": "UNSUPPORTED", "reason": f"Unknown intent: {intent}"}
        )

    # ========== RAG WORKFLOW ==========
    def _plan_rag(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan pure RAG workflow"""
        slots = nlu_result.slots
        # Domain-specific retrieval hints
        if slots.get("domain") == "field_standards":
            keywords = "field dimensions standards age group requirements length width pitching distance soccer baseball softball"
        else:
            keywords = "mowing maintenance safety equipment frequency inspection checklist"

        tool_chain = [
            {
                "tool": "kb_retrieve",
                "args": {"query": keywords, "top_k": 5}
            },
            {
                "tool": "sop_extract",
                "args": {"schema": ["steps", "materials", "tools", "safety"]}
            }
        ]
        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=[],
            metadata={"workflow": "RAG", "domain": slots.get("domain")}
        )

    # ========== SQL WORKFLOW ==========
    def _plan_sql(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan pure SQL workflow"""
        slots = nlu_result.slots
        query = nlu_result.raw_query

        # Step 1: Template routing
        template = self._route_sql_template(query, slots)
        if not template:
            # Not supported: no template match for this SQL intent
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "SQL", "status": "UNSUPPORTED", "reason": "NO_TEMPLATE_MATCH"}
            )

        # Step 2: Template lookup
        template_config = self.sql_templates.get(template)
        if not template_config:
            # Not supported: routed template not registered
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "SQL", "status": "UNSUPPORTED", "reason": f"UNREGISTERED_TEMPLATE:{template}"}
            )

        # Step 3: Build params & validate
        params = template_config["builder"](slots)
        clarifications = self._validate_params(template, params, template_config)

        tool_chain = []
        if not clarifications:
            tool_chain.append({
                "tool": "sql_query_rag",
                "args": {"template": template, "params": params}
            })

        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=clarifications,
            metadata={"workflow": "SQL", "template": template, "status": "OK" if not clarifications else "NEEDS_CLARIFICATION"}
        )

    # ========== RAG+SQL WORKFLOW ==========
    def _plan_rag_sql(self, nlu_result: NLUResult) -> ExecutionPlan:
        """
        Entry for hybrid RAG+SQL workflow.
        Dispatches by domain to keep per-domain logic clear.
        """
        domain = nlu_result.slots.get("domain")

        if domain == "field_dimension":
            return self._plan_rag_sql_fields(nlu_result)
        if domain == "activity":
            return self._plan_rag_sql_activity(nlu_result)
        # default / mowing and other SQL-toolable ops
        return self._plan_rag_sql_mowing(nlu_result)

    def _plan_rag_sql_mowing(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan hybrid RAG+SQL workflow for mowing domain"""
        slots = nlu_result.slots
        query = nlu_result.raw_query

        # SQL template routing first (if unsupported, we bail out)
        template = self._route_sql_template(query, slots)
        if not template:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "RAG+SQL", "status": "UNSUPPORTED", "reason": "NO_TEMPLATE_MATCH"}
            )

        template_config = self.sql_templates.get(template)
        if not template_config:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "RAG+SQL", "status": "UNSUPPORTED", "reason": f"UNREGISTERED_TEMPLATE:{template}"}
            )

        params = template_config["builder"](slots)
        clarifications = self._validate_params(template, params, template_config)

        # RAG retrieval keywords for context (kept lightweight)
        rag_keywords = "mowing maintenance cost frequency staffing inspection policy standards"

        tool_chain: List[Dict[str, Any]] = []
        if not clarifications:
            tool_chain = [
                {"tool": "kb_retrieve", "args": {"query": rag_keywords, "top_k": 3}},
                {"tool": "sql_query_rag", "args": {"template": template, "params": params}}
            ]

        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=clarifications,
            metadata={
                "workflow": "RAG+SQL",
                "template": template,
                "status": "OK" if not clarifications else "NEEDS_CLARIFICATION",
                "explanation_requested": bool(slots.get("explanation_requested"))
            }
        )
    
    def _plan_rag_sql_activity(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan hybrid RAG+SQL workflow for activity domain"""
        slots = nlu_result.slots
        query = nlu_result.raw_query

        # SQL template routing first (if unsupported, bail out)
        template = self._route_sql_template(query, slots)
        if not template:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "RAG+SQL", "status": "UNSUPPORTED", "reason": "NO_TEMPLATE_MATCH"}
            )

        template_config = self.sql_templates.get(template)
        if not template_config:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[],
                metadata={"workflow": "RAG+SQL", "status": "UNSUPPORTED", "reason": f"UNREGISTERED_TEMPLATE:{template}"}
            )

        params = template_config["builder"](slots)
        clarifications = self._validate_params(template, params, template_config)

        # Activity-specific RAG context
        base_kw = slots.get("activity_name") or "maintenance activity"
        rag_keywords = f"{base_kw} frequency interval guidelines maintenance"
        kb_filters = {"category": "activity"}

        tool_chain: List[Dict[str, Any]] = []
        if not clarifications:
            tool_chain = [
                {"tool": "kb_retrieve", "args": {"query": rag_keywords, "top_k": 3, "filters": kb_filters}},
                {"tool": "sql_query_rag", "args": {"template": template, "params": params}}
            ]

        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=clarifications,
            metadata={
                "workflow": "RAG+SQL",
                "template": template,
                "status": "OK" if not clarifications else "NEEDS_CLARIFICATION",
                "explanation_requested": bool(slots.get("explanation_requested")),
                "domain": "activity",
            }
        )
    
    def _plan_rag_sql_fields(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan hybrid RAG+SQL workflow for field dimension queries"""
        slots = nlu_result.slots
        query = nlu_result.raw_query
        
        # RAG component: retrieve context
        rag_keywords = self._build_rag_keywords(slots)
        
        # SQL component: determine template and params
        template = self._route_sql_template(query, slots)
        template_config = self.sql_templates.get(template)
        
        if not template_config:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=[f"Unknown SQL template: {template}"]
            )
        
        
        param_builder = template_config["builder"]
        params = param_builder(nlu_result.raw_query)
        slots["field_type"] = "diamond" if template == "field_dimension.diamond" else "rectangular"
        # Validate parameters
        clarifications = self._validate_params(template, params, template_config)
        if clarifications:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=clarifications
            )
        tool_chain = [
            {
                "tool": "kb_retrieve",
                "args": {
                    "query": rag_keywords,
                    "top_k": 3
                }
            },
            {
                "tool": "sql_query_rag",
                "args": {
                    "template": template, 
                    "params": params
                }
            }
        ]
        
        print(f"[Planner] RAG+SQL plan: template={template}")
        
        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications="",
            metadata={"workflow": "RAG+SQL", "template": template, "status": "OK"}
        )
    # ========== CV WORKFLOW ==========
    def _plan_cv(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan pure CV workflow"""
        slots = nlu_result.slots
        image_uri = slots.get("image_uri")

        if not image_uri:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=["Please upload an image for computer vision analysis"],
                metadata={"workflow": "CV", "status": "NEEDS_CLARIFICATION"}
            )

        tool_chain = [
            {
                "tool": "cv_assess_rag",
                "args": {
                    "image_uri": image_uri,
                    "topic_hint": "turf wear disease inspection guidelines"
                }
            }
        ]
        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=[],
            metadata={"workflow": "CV", "status": "OK"}
        )

    # ========== RAG+CV WORKFLOW ==========
    def _plan_rag_cv(self, nlu_result: NLUResult) -> ExecutionPlan:
        """Plan hybrid RAG+CV workflow"""
        slots = nlu_result.slots
        image_uri = slots.get("image_uri")

        if not image_uri:
            return ExecutionPlan(
                tool_chain=[],
                clarifications=["Please upload an image for analysis"],
                metadata={"workflow": "RAG+CV", "status": "NEEDS_CLARIFICATION"}
            )

        tool_chain = [
            {"tool": "kb_retrieve", "args": {"query": "turf inspection maintenance repair standards", "top_k": 3}},
            {"tool": "cv_assess_rag", "args": {"image_uri": image_uri, "topic_hint": "turf wear disease inspection guidelines"}}
        ]
        return ExecutionPlan(
            tool_chain=tool_chain,
            clarifications=[],
            metadata={"workflow": "RAG+CV", "status": "OK"}
        )

    # ========== SQL TEMPLATE ROUTING ==========
    def _route_sql_template(self, query: str, slots: Dict[str, Any]) -> Optional[str]:
        """
        Determine which SQL template to use based on query patterns.
        Return None when no supported template is identified (so executor can say "not supported").
        """
        lowq = (query or "").lower().strip()
        domain = slots.get("domain")
        
        # Handle field dimension domain
        if domain == "field_dimension":
            if slots.get("sport") in ["soccer", "cricket", "gaelic football", "cfl", "nfl", "rugby", "ultimate frisbee", "lacrosse", "frisbee", "ultimate"]:
                return "field_dimension.rectangular"
            return "field_dimension.diamond"
        
        # Handle mowing domain
        if domain == "mowing":
            # Pattern 1: Superlative queries (highest, top, most expensive)
            if any(k in lowq for k in ["highest", "top", "max", "most expensive"]) and "cost" in lowq:
                return "mowing.labor_cost_month_top1"

            # Pattern 2: Recent/last queries
            if any(k in lowq for k in ["last", "recent", "latest", "when was"]) and any(k in lowq for k in ["mow", "mowing"]):
                return "mowing.last_mowing_date"

            # Pattern 3: Trend queries
            if "trend" in lowq and "cost" in lowq:
                return "mowing.cost_trend"

            # Pattern 4: Range queries (from X to Y)
            if re.search(r'\bfrom\s+\w+\s+to\s+\w+', lowq) and "cost" in lowq:
                return "mowing.cost_trend"

            # Pattern 5: Comparison queries
            if any(k in lowq for k in ["compare", "across", "all parks"]) and "cost" in lowq:
                return "mowing.cost_by_park_month"

            # Pattern 6: Breakdown queries
            if any(k in lowq for k in ["breakdown", "detail", "break down"]) and "cost" in lowq:
                return "mowing.cost_breakdown"

            # Pattern 7: By-park phrasing
            if "by park" in lowq or "each park" in lowq:
                return "mowing.cost_by_park_month"

            # Pattern 8: Parks having the least mowing cost for a period
            if "least" in lowq and "mowing cost" in lowq:
                return "mowing.cost_by_park_least_per_sqft"
        # Handle activity domain
        if domain == "activity":
            # Pattern 11: Maintenance due within a time window
            if slots.get("weeks_ahead") or ("need" in lowq and "week" in lowq):
                return "activity.maintenance_due_window"
            # Pattern 9: Activity/maintenance cost within a park and date range
            if "cost" in lowq and "in" in lowq and (("from" in lowq and " to " in lowq) or "between" in lowq):
                return "activity.cost_by_location_range"
            # Pattern 10: Last activity date for a park
            if any(k in lowq for k in ["last", "recent", "latest", "when was"]):
                return "activity.last_activity_date"

        # No supported template matched
        return None

    # ========== SQL PARAMETER BUILDERS ==========
    def _build_top_cost_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for top cost query"""
        return {"month": slots.get("month"), "year": slots.get("year"), "park_name": slots.get("park_name")}

    def _build_trend_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for trend analysis query"""
        return {
            "year": slots.get("range_year") or slots.get("year"),
            "start_month": slots.get("start_month") or slots.get("month") or 1,
            "end_month": slots.get("end_month") or slots.get("month") or 12,
            "park_name": slots.get("park_name"),
        }

    def _build_last_date_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for last mowing date query"""
        return {"park_name": slots.get("park_name")}

    def _build_cost_by_park_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for cost-by-park query"""
        return {"month": slots.get("month"), "year": slots.get("year"), "park_name": slots.get("park_name")}

    def _build_breakdown_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for cost breakdown query"""
        return {"month": slots.get("month"), "year": slots.get("year"), "park_name": slots.get("park_name")}
    def _build_cost_least_per_sqft_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for least cost per sqft query"""
        return {
            "month1": slots.get("month1"), 
            "month2": slots.get("month2"), 
            "year1": slots.get("year1"),
            "year2": slots.get("year2")
        }
    def _build_activity_cost_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for park activity cost range query"""
        return {
            "park_name": slots.get("park_name"),
            "month1": slots.get("month1"),
            "month2": slots.get("month2"),
            "year1": slots.get("year1"),
            "year2": slots.get("year2"),
            "activity_name": slots.get("activity_name")
        }
    def _build_last_activity_date_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for last activity date query"""
        return {
            "park_name": slots.get("park_name"),
            "activity_name": slots.get("activity_name")
        }
    def _build_activity_due_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for maintenance due-within-window query"""
        return {
            "activity_name": slots.get("activity_name"),
            "weeks_ahead": slots.get("weeks_ahead"),
            "park_name": slots.get("park_name")
        }
    def _build_field_dimension_params_diamond(self, query: str) -> Dict[str, Any]:
        # Implementation for building parameters for diamond field dimension queries
        t = query
        diamond_fields = ['Balaclava Diamond SE', 'Balaclava Diamond SW', 'Beaconsfield Diamond SW', 'Bobolink Diamond EC', 'Bobolink Diamond NE', 'Bobolink Diamond NW', 'Bobolink Diamond SW', 'Braemar Diamond SE', 'Stanley Park Brockton Diamond SW', 'Carnarvon Diamond EC', 'Carnarvon Diamond NE', 'Carnarvon Diamond NW', 'Carnarvon Diamond SW', 'Chaldecott Diamond C', 'Chaldecott Diamond NW', 'Champlain Diamond NE', 'Champlain Diamond SW', 'China Creek Diamond NE', 'China Creek Diamond NW', 'Clark Diamond S', 'Clinton Diamond NW', 'Collingwood Diamond NW', 'Columbia Diamond NE', 'Columbia Diamond SE', 'Columbia Diamond WC', 'Connaught Diamond NE', 'Connaught Diamond NW', 'Connaught Diamond SE', 'Connaught Diamond SC', 'Douglas Field W Diamond NW', 'Douglas Field E Diamond SC', 'Douglas Field E Diamond SE', 'Douglas Field W Diamond SW', 'Earles Diamond SW', 'Elm Diamond NW', 'Falaise Diamond NE', 'Falaise Diamond S', 'Gaston Diamond NW', 'Gordon Diamond SE Centre', 'Gordon Diamond NE', 'Gordon Diamond SE', 'Gordon Diamond SW', 'Gordon Diamond NW Centre', 'Hastings Diamond SE', 'Hastings Diamond SW', 'Hillcrest Diamond NE', 'Hillcrest Diamond NW', 'Hillcrest Diamond SE - Challenger', 'Hillcrest Diamond SW', 'Jericho Diamond N', 'John Hendry Diamond EC', 'John Hendry Diamond EN', 'John Hendry Diamond NW', 'John Hendry Diamond ES', 'John Hendry Diamond SW', 'John Hendry Field Gravel Diamond', 'Jonathan Rogers Diamond NW', 'Kensington Diamond NSE', 'Kensington Diamond NNW', 'Kensington Diamond SSE', 'Kensington Diamond SNW', 'Kerrisdale Diamond NW', 'Kerrisdale Diamond SE', 'Killarney Diamond NW', 'Killarney Diamond C', 'Killarney Diamond EC', 'Killarney Diamond NE', 'Locarno Beach Diamond ', 'Maple Grove Diamond NE', 'Maple Grove Diamond SW', 'Mcbride Diamond SE', 'Mcbride Diamond SW', 'Memorial South Diamond C', 'Memorial South Diamond SC', 'Memorial South Diamond NE', 'Memorial South Diamond NW', 'Memorial South Diamond SE', 'Memorial South Diamond SE', 'Memorial West Diamond C', 'Memorial West Diamond E', 'Memorial West Diamond W', 'Moberly Diamond SE', 'Moberly Diamond SW', 'Montgomery Diamond NW', 'Montgomery Diamond NE', 'Montgomery Diamond SE', 'Montgomery Diamond SW', 'Nanaimo Diamond NC', 'Nanaimo Diamond NE', 'Nanaimo Diamond SE', 'Nanaimo Diamond SW', 'Norquay Diamond NE', 'Norquay Diamond SW', 'Oak Diamond N', 'Oak Diamond NE', 'Oak Diamond S', 'Oak Diamond SE', 'Quilchena Diamond C', 'Quilchena Diamond SE', 'Riley Diamond SW', 'Ross Diamond NE', 'Rupert Diamond C', 'Shann Diamond SE', 'Strathcona Diamond C', 'Strathcona Diamond NW', 'Strathcona Diamond SW', 'Sunrise Diamond SE', 'Templeton Diamond SE', 'Tisdall Diamond NE', 'Tisdall Diamond SW', 'Trafalgar Diamond SE', 'Trafalgar Diamond SW', 'Trafalgar Diamond W', 'West Point Grey Diamond NE', 'West Point Grey Diamond SE', 'West Point Grey Diamond SW', 'Woodland Diamond NW', 'None']
        match = process.extractOne(t, diamond_fields, score_cutoff=70)
        return {"field_name": match[0] if match else None }
    def _build_field_dimension_params_rectangular(self, query: str) -> Dict[str, Any]:
        # Implementation for building parameters for diamond field dimension queries
        
        t = query
        rectangle_fields = ['Adanac Field NW', 'Adanac Field SE', 'Adanac Field Summer', 'Adanac Field SW', 'Andy Livingstone  Artificial Field E', 'Andy Livingstone  Artificial Field W', 'Balaclava Field Oval', 'Balaclava Field SW', 'Beaconsfield Field Gravel', 'Beaconsfield Field SE', 'Beaconsfield Field SW', 'Bobolink Field NW', 'Bobolink Field SE', 'Bobolink Field SW', 'Braemar Field C', 'Brewers Field C', 'Carnarvon Field E Summer', 'Carnarvon Field N', 'Carnarvon Field S', 'Carnarvon Field WC Summer', 'Cartier Grass Area', 'Chaldecott Field NW', 'Chaldecott Field SW', 'Champlain Field C', 'Charleson Field C', 'China Creek rth Field C', 'Clark Field C', 'Clinton Field Gravel', 'Clinton Field NW', 'Clinton Field SW', 'Collingwood Field C', 'Columbia Field C', 'Columbia Field S', 'Connaught Cricket Pitch E', 'Connaught Field C', 'Connaught Field E', 'Connaught Field NW', 'Connaught Field W', 'David Lam Field C', 'Douglas Field E', 'Douglas Field E - Cricket Pitch', 'Douglas Field E North', 'Douglas Field W', 'Earles Field C', 'Elm Field C', 'Empire Artificial Field N', 'Empire Artificial Field S', 'Eric Hamber High School Synthetic Turf', 'Falaise Field N', 'Garden Field C', 'Gaston Field S', 'General Brock Field N', 'George Field C', 'Gordon Field NE', 'Gordon Field NW', 'Gordon Field SE', 'Granville Field C', 'Hillcrest Field E', 'Hillcrest Field Mini NW', 'Hillcrest Field N', 'Hillcrest Field S', 'Jericho Field E', 'Jericho Grass - E Festival Area', 'Jericho West Artificial Field', 'John Hendry Field Gravel', 'John Hendry Field NE', 'John Hendry Field NW', 'Jonathan Rogers Field C', 'Jones Field C', 'Kensington Field N', 'Kensington Field S', 'Kerrisdale ES Field Gravel N', 'Kerrisdale ES Field Gravel S', 'Kerrisdale Field C', 'Killarney Field C', 'Killarney Field EC', 'Killarney Field Gravel', 'Killarney Field Oval', 'Killarney Field SE ', 'Kingcrest Field C', 'Locar Beach Field Gravel', 'Locarno Beach Field C', 'Maclean Field C', 'McBride Field C', 'McSpadden Field C', 'Memorial South Artifical Field', 'Memorial South Field NE', 'Memorial South Field NW', 'Memorial South Field Oval', 'Memorial South Field SE', 'Memorial West Field C', 'Memorial West Field SE', 'Moberly Field N', 'Moberly Field S', 'Montgomery Field E', 'Montgomery Field W', 'Musqueam Field E', 'Musqueam Field NW', 'Musqueam Field W', 'Nanaimo Field E', 'Nanaimo Field W', 'New Brighton Field N', 'Norquay Field S', 'Oak Field E', 'Oak Field Gravel', 'Oak Field W', 'Oak Meadows Field C', 'Point Grey HS Artificial Field', 'Prince Edward Field C', 'Prince of Wales Field C', 'Queen Elizabeth ES Field Gravel', 'Quilchena Field NW', 'Quilchena Field SE', 'Renfrew Field C', 'Riley Field C', 'Robson Field C', 'Ross Field C', 'Rupert Field C', 'Rupert Field S', 'Shannon Field C', 'Slocan Field C', 'Slocan Field SW', 'Stanley Park Brockton Cricket Field N', 'Stanley Park Brockton Cricket Field S', 'Stanley Park Brockton Field Oval', 'Stanley Park Brockton Field SW', 'Strathcona Field EC', 'Strathcona Field Gravel', 'Strathcona Field Oval', 'Strathcona Field W', 'Sunrise Field C', 'Sunset/Henderson Field C', 'Templeton Field N', 'Templeton Field S', 'Tisdall Field N', 'Tisdall Field S', 'Trafalgar Cricket Pitch C', 'Trafalgar Field C', 'Trafalgar Field E', 'Trafalgar Field NW', 'Trafalgar Field SC', 'Trillium Artificial Field E', 'Trillium Artificial Field W', 'Van Tech HS Artificial Field', 'West Point Grey Field N', 'West Point Grey Field S', 'Wina Field C', 'Wina Field N', 'Wina Field S', 'Woodland Field N', 'Woodland Field S', 'None']
        match = process.extractOne(t, rectangle_fields, score_cutoff=40)
        print(f"[Planner] Field dimension match: {match}")
        return {"field_name": match[0] if match else None }
    # ========== PARAMETER VALIDATION ==========
    def _validate_params(self, template: str, params: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """
        Validate required parameters and generate clarification prompts.
        If required params are missing -> return clarifications (NOT unsupported).
        """
        clarifications: List[str] = []
        required = config.get("required", [])
        missing = [k for k in required if not params.get(k)]

        if not missing:
            return clarifications

        # Template-specific clarification copy
        if template == "mowing.cost_trend":
            clarifications.append("Which time period would you like to see? (e.g., from January to June, and year)")
        elif template in ["mowing.cost_by_park_month", "mowing.cost_breakdown", "mowing.labor_cost_month_top1"]:
            clarifications.append("Which month and year would you like to query?")
        # mowing.last_mowing_date has no required fields; nothing to add
        elif template == "mowing.cost_by_park_least_per_sqft":
            clarifications.append("Please specify the start month, end month, start year, and end year for the period.")
        elif template == "activity.cost_by_location_range":
            clarifications.append("Please provide the park name along with start and end month/year (e.g., from March 2024 to May 2024).")
        elif template == "activity.last_activity_date":
            clarifications.append("Please provide the park name for which you want to know the last activity date.")
        elif template == "activity.maintenance_due_window":
            clarifications.append("Please specify the maintenance activity (e.g., mowing) and the horizon in weeks.")
        elif template in ["field_dimension.rectangular", "field_dimension.diamond"]:
            clarifications.append("Please specify the field name or sport for which you want the dimensions.")
        return clarifications
    
    def _build_rag_keywords(self, slots: Dict[str, Any]) -> str:
        """Build RAG retrieval keywords based on domain and query"""
        res = ""
        for key, value in slots.items():
            if key != "domain" and value is not None:
                res += f"{key}:{value} " 
        return res.strip()
