"""
Retrieval Module: The 'Search Engine' of Policy Explainer.
This module uses Multi-Query Retrieval to gather the most relevant document
chunks for each specific section of your Policy Summary.
"""

from typing import Any, Final
from backend import storage

# --- Constants ---

# Canonical policy sections: These align with your SectionName Literal in schemas.py.
# Using Final ensures these cannot be changed accidentally during runtime.
# Updated retrieval.py keys to match schemas.py Literals
CORE_SECTIONS: Final[tuple[str, ...]] = (
    "Plan Snapshot",
    "Cost Summary",
    "Summary of Covered Services", # Match Schema
    "Administrative Conditions",
    "Exclusions & Limitations",
    "Claims, Appeals & Member Rights", # Match Schema
)

# Multi-Query Mapping:
# NLP Strategy: Instead of searching for "Cost Summary", we search for 5 synonyms.
# This significantly improves 'Recall' because insurance documents use varied jargon.
SECTION_QUERIES: Final[dict[str, tuple[str, ...]]] = {
    "Plan Snapshot": (
        "plan name and type",
        "summary of benefits overview",
        "plan overview and key features",
    ),
    "Cost Summary": (
        "deductible amount and when it applies",
        "copay and coinsurance",
        "out of pocket maximum OOP",
        "annual deductible",
        "cost sharing requirements",
    ),
    "Summary of Covered Services": (
        "what is covered",
        "covered benefits and services",
        "coverage details",
        "covered medical services",
        "benefits included in plan",
    ),
    "Administrative Conditions": (
        "prior authorization",
        "referrals required",
        "administrative requirements",
    ),
    "Exclusions & Limitations": (
        "exclusions not covered",
        "limitations and restrictions",
        "what is not covered",
    ),
    "Claims, Appeals & Member Rights": (
        "how to file a claim",
        "appeals and grievances",
        "member rights and responsibilities",
    ),
}

TOP_K_PER_QUERY = 4  # Number of chunks to grab per sub-query
MAX_CHUNKS_SECTION = 18  # Global limit to prevent passing too much text to the LLM


def retrieve_for_section(
    doc_id: str,
    section_name: str,
    top_k_per_query: int = TOP_K_PER_QUERY,
    max_chunks: int = MAX_CHUNKS_SECTION,
) -> list[dict[str, Any]]:
    """
    Orchestrates the retrieval for a specific section.
    1. Fetches sub-queries from SECTION_QUERIES.
    2. Queries the Vector Store (storage.py).
    3. Dedupes chunks (one chunk might match multiple sub-queries).
    4. Sorts by page number for logical context.
    """
    queries = SECTION_QUERIES.get(section_name)
    if not queries:
        return []

    seen: dict[str, dict[str, Any]] = {}

    for q in queries:
        if not q.strip():
            continue

        # storage.query connects to your Vector Database (e.g., ChromaDB)
        hits = storage.query(doc_id, q.strip(), top_k=top_k_per_query)

        for h in hits:
            cid = h.get("chunk_id") or ""
            if not cid:
                continue

            # --- Deduping & Distance Logic ---
            # If we see the same chunk twice, we keep the one with the
            # smaller 'distance' (higher similarity).
            if cid not in seen:
                out = dict(h)
                out["section"] = section_name
                seen[cid] = out
            else:
                d = h.get("distance")
                # Lower distance = closer match in vector space
                if d is not None and seen[cid].get("distance") is not None and d < seen[cid]["distance"]:
                    out = dict(h)
                    out["section"] = section_name
                    seen[cid] = out

    # --- Sorting for Context ---
    # We sort by page_number so the LLM reads the document in the order it was written.
    # This helps the LLM understand 'flow' and avoids confusion.
    ordered = sorted(
        seen.values(),
        key=lambda x: (x.get("page_number", 0), x.get("chunk_id", ""))
    )

    # Cap the results to prevent 'Lost in the Middle' LLM issues
    return ordered[:max_chunks]