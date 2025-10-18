# nlu.py - Natural Language Understanding Layer
# Responsibility: Intent classification + Slot filling ONLY
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from sentence_transformers import SentenceTransformer, util

# ========== MODEL INITIALIZATION ==========
_ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ========== INTENT PROTOTYPES ==========
INTENT_PROTOTYPES = {
    "RAG+SQL_tool": [
        "Which park had the highest total mowing labor cost in March 2025?",
        "Show me the most expensive park for mowing in April",
        "What was the top mowing cost by location last month?",
    ],
    "RAG": [
        "What are the mowing steps and safety requirements?",
        "How do I maintain the turf properly?",
        "What equipment do I need for mowing?",
        "Tell me the standard operating procedures for mowing",
        "What are the dimensions for U15 soccer?",
        "Show me baseball field requirements for U13",
        "What's the pitching distance for female softball U17?",
    ],
    "SQL_tool": [
        "How did total mowing costs trend from April to June 2025?",
        "Compare mowing costs across all parks in March 2025",
        "When was the last mowing at Cambridge Park?",
        "Show me all parks ranked by mowing cost",
    ],
    "RAG+CV_tool": [
        "Here is a picture. What's the estimated labour cost to repair the turf?",
        "Check this image and tell me if the grass needs mowing",
        "Analyze this field photo and estimate repair costs",
    ],
    "CV_tool": [
        "Check this image and assess turf condition.",
        "Analyze this photo of the field",
        "What do you see in this image?",
    ],
}

# Pre-encode all prototypes
_PROT_TEXTS = []
_PROT_LABELS = []
for label, samples in INTENT_PROTOTYPES.items():
    for sample in samples:
        _PROT_TEXTS.append(sample)
        _PROT_LABELS.append(label)
_PROT_EMB = _ENCODER.encode(_PROT_TEXTS, normalize_embeddings=True)

# ========== ENTITY EXTRACTION UTILITIES ==========
_MONTHS = {
    m: i for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"],
        start=1
    )
}
_MONTH_ABBR = {k[:3]: v for k, v in _MONTHS.items()}


def _normalize(text: Optional[str]) -> str:
    """Normalize text to lowercase and strip whitespace"""
    return (text or "").strip().lower()


def _parse_month_year(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract month and year from text"""
    t = _normalize(text).replace(",", " ")
    
    # Pattern: YYYY-MM
    m = re.search(r"\b(20\d{2})-(\d{1,2})\b", t)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return month, year
    
    # Pattern: Full month name
    for name, idx in _MONTHS.items():
        if re.search(rf"\b{name}\b", t):
            month = idx
            year_match = re.search(r"\b(20\d{2})\b", t)
            year = int(year_match.group(1)) if year_match else None
            return month, year
    
    # Pattern: Abbreviated month
    for abbr, idx in _MONTH_ABBR.items():
        if re.search(rf"\b{abbr}\b", t):
            month = idx
            year_match = re.search(r"\b(20\d{2})\b", t)
            year = int(year_match.group(1)) if year_match else None
            return month, year
    
    # Pattern: Year only
    year_match = re.search(r"\b(20\d{2})\b", t)
    if year_match:
        return None, int(year_match.group(1))
    
    return None, None


def _parse_month_range(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract month range from text
    Returns: (start_month, end_month, year)
    """
    t = _normalize(text)
    
    # Pattern: "from April to June 2025"
    m = re.search(
        r"(?:from|between)\s+([a-zA-Z]+)\s+(?:to|and)\s+([a-zA-Z]+)\s*(20\d{2})?",
        t
    )
    if m:
        m1, m2 = m.group(1).lower(), m.group(2).lower()
        year = int(m.group(3)) if m.group(3) else None
        
        def to_month(s: str) -> Optional[int]:
            if s in _MONTHS:
                return _MONTHS[s]
            return _MONTH_ABBR.get(s[:3])
        
        return to_month(m1), to_month(m2), year
    
    # Pattern: "in June 2025" (single month as range)
    for name, idx in _MONTHS.items():
        if re.search(rf"\bin\s+{name}\b", t):
            year_match = re.search(r"\b(20\d{2})\b", t)
            year = int(year_match.group(1)) if year_match else None
            return idx, idx, year
    
    for abbr, idx in _MONTH_ABBR.items():
        if re.search(rf"\bin\s+{abbr}\b", t):
            year_match = re.search(r"\b(20\d{2})\b", t)
            year = int(year_match.group(1)) if year_match else None
            return idx, idx, year
    
    return None, None, None


def _parse_park_name(text: str) -> Optional[str]:
    """Extract park name from text"""
    t = _normalize(text)
    
    # Pattern: "in/at XXX Park"
    m = re.search(r"(?:in|at|for)\s+([a-z][a-z\s\-\&]+(?:park|pk))\b", t)
    if m:
        park_raw = m.group(1).strip()
        park_clean = park_raw.replace(" park", "").replace(" pk", "").strip()
        return park_clean.title()
    
    # Known park names
    known_parks = [
        "alice town", "cambridge", "garden", "grandview",
        "mcgill", "mcspadden", "mosaic creek", "cariboo",
        "john hendry", "hastings", "new brighton"
    ]
    for park in known_parks:
        if park in t:
            return park.title()
    
    return None


def _detect_domain(text: str) -> str:
    """Detect query domain: mowing / field_standards / generic"""
    t = _normalize(text)
    
    if any(k in t for k in ["mowing", "mow", "turf", "grass", "lawn"]):
        return "mowing"
    
    if any(k in t for k in ["soccer", "baseball", "softball", "cricket", 
                            "football", "rugby", "field", "dimensions", 
                            "pitching", "u10", "u11", "u12", "u13", 
                            "u14", "u15", "u16", "u17", "u18"]):
        return "field_standards"
    
    return "generic"


# ========== NLU RESULT DATACLASS ==========
@dataclass
class NLUResult:
    """Output from NLU layer"""
    intent: str
    confidence: float
    slots: Dict[str, Any]
    raw_query: str


# ========== MAIN NLU FUNCTION ==========
def parse_intent_and_slots(
    text: str,
    image_uri: Optional[str] = None
) -> NLUResult:
    """
    Parse user input to extract intent and slots
    
    Args:
        text: User query text
        image_uri: Optional image URI
    
    Returns:
        NLUResult with intent classification and extracted entities
    """
    query = text.strip()
    
    # ========== STEP 1: Intent Classification ==========
    query_embedding = _ENCODER.encode([query], normalize_embeddings=True)
    similarities = util.cos_sim(query_embedding, _PROT_EMB).cpu().tolist()[0]
    
    # Get best matching prototype
    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    intent = _PROT_LABELS[best_idx]
    confidence = float(similarities[best_idx])
    
    print(f"[NLU] Initial intent: {intent} (confidence: {confidence:.3f})")
    
    # ========== STEP 2: Multimodal Intent Correction ==========
    if image_uri:
        print(f"[NLU] Image detected, adjusting intent to include CV")
        if "CV" not in intent:
            intent = "RAG+CV_tool" if intent in ["RAG", "SQL_tool", "RAG+SQL_tool"] else "CV_tool"
    else:
        if "CV" in intent:
            print(f"[NLU] No image but CV intent detected, downgrading to RAG")
            intent = "RAG"
    
    # ========== STEP 3: Keyword-Based Intent Refinement ==========
    lowq = _normalize(query)
    informational_keywords = [
        "steps", "procedure", "safety", "manual", "how to", "sop",
        "dimensions", "requirements", "standards", "what are", "show me", "tell me"
    ]
    
    if any(k in lowq for k in informational_keywords):
        if intent not in ["RAG", "RAG+CV_tool"]:
            print(f"[NLU] Informational keywords detected, forcing to RAG")
            intent = "RAG" if not image_uri else "RAG+CV_tool"
    
    # ========== STEP 4: Slot Filling (NER) ==========
    domain = _detect_domain(query)
    month, year = _parse_month_year(query)
    start_month, end_month, range_year = _parse_month_range(query)
    park_name = _parse_park_name(query)
    
    slots = {
        "domain": domain,
        "month": month,
        "year": year,
        "start_month": start_month,
        "end_month": end_month,
        "range_year": range_year,
        "park_name": park_name,
        "image_uri": image_uri,
    }
    
    # ========== LOGGING ==========
    print(f"[NLU] Final intent: {intent}")
    print(f"[NLU] Domain: {domain}")
    print(f"[NLU] Extracted slots: {slots}")
    
    return NLUResult(
        intent=intent,
        confidence=round(confidence, 3),
        slots=slots,
        raw_query=query
    )