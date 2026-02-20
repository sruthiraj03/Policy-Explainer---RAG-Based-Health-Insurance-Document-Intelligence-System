"""Retrieval: section list, section queries, and section-specific vector search."""

from typing import Any, Final

from backend import storage

# Canonical policy sections (SYSTEM_SPEC)
CORE_SECTIONS: Final[tuple[str, ...]] = (
    "Plan Snapshot",
    "Cost Summary",
    "Covered Services",
    "Administrative Conditions",
    "Exclusions & Limitations",
    "Claims/Appeals & Member Rights",
)

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
    "Covered Services": (
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
    "Claims/Appeals & Member Rights": (
        "how to file a claim",
        "appeals and grievances",
        "member rights and responsibilities",
    ),
}

TOP_K_PER_QUERY = 4
MAX_CHUNKS_SECTION = 18


def retrieve_for_section(
    doc_id: str,
    section_name: str,
    top_k_per_query: int = TOP_K_PER_QUERY,
    max_chunks: int = MAX_CHUNKS_SECTION,
) -> list[dict[str, Any]]:
    """Return relevant chunks for a section by running section-specific queries; merge and dedupe by chunk_id."""
    queries = SECTION_QUERIES.get(section_name)
    if not queries:
        return []
    seen: dict[str, dict[str, Any]] = {}
    for q in queries:
        if not q.strip():
            continue
        hits = storage.query(doc_id, q.strip(), top_k=top_k_per_query)
        for h in hits:
            cid = h.get("chunk_id") or ""
            if not cid:
                continue
            if cid not in seen:
                out = dict(h)
                out["section"] = section_name
                seen[cid] = out
            else:
                d = h.get("distance")
                if d is not None and seen[cid].get("distance") is not None and d < seen[cid]["distance"]:
                    out = dict(h)
                    out["section"] = section_name
                    seen[cid] = out
    ordered = sorted(seen.values(), key=lambda x: (x.get("page_number", 0), x.get("chunk_id", "")))
    return ordered[:max_chunks]
