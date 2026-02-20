"""Q&A: grounded RAG, section deep-dive, scenario generator, preventive helper."""

import json
import re
from typing import Any

from openai import OpenAI

from backend import storage
from backend.config import get_settings
from backend.evaluation import confidence_for_qa, validate_qa_response
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.schemas import (
    Citation,
    NOT_FOUND_MESSAGE,
    PageCitation,
    QA_RESPONSE_DISCLAIMER,
    QAResponseOutput,
    PreventiveQAResponseOutput,
    SectionDetailQAResponseOutput,
    ScenarioQAResponseOutput,
    ScenarioStepOutput,
)
from backend.summarization import summarize_section
from backend.utils import load_terminology_map, normalize_text

NOT_FOUND_ANSWER = "Not found in this document."
DEFAULT_TOP_K = 6

# --- Section deep-dive ---
DETAIL_INTENT_PATTERNS = [
    r"more\s+detail\s+about", r"in\s+more\s+detail", r"deeper\s+summary\s+of",
    r"explain\s+.+\s+in\s+more\s+detail", r"detailed\s+summary\s+of", r"deep\s+dive\s+(?:into|on)",
    r"more\s+detail\s+on", r"expand\s+on\s+(?:the\s+)?", r"tell\s+me\s+more\s+about\s+(?:the\s+)?",
]

# --- Scenario ---
SCENARIO_HEADER = "Example Scenario (Hypothetical – Based on Policy Terms)"
SCENARIO_TRIGGER_PHRASES = ["what would happen if", "example scenario", "how much would i pay if", "if i visit the er", "walk me through", "example of", "scenario:"]
SCENARIO_TYPES = {
    "ER visit": ["er", "emergency room", "emergency", "ed visit"],
    "Specialist visit": ["specialist", "specialty", "referral"],
    "Prescription fill": ["prescription", "pharmacy", "medication", "drug"],
    "Therapy sessions": ["therapy", "mental health", "counseling", "behavioral"],
    "Maternity delivery": ["maternity", "delivery", "childbirth", "prenatal"],
}

# --- Preventive ---
PREVENTIVE_KEYWORDS = ("vaccine", "vaccination", "immunization", "annual exam", "screening", "preventive", "preventative", "physical exam", "wellness visit", "mammogram", "colonoscopy", "pap smear", "flu shot", "checkup", "preventive care")
GENERAL_PREVENTIVE_GUIDANCE = "Preventive services and coverage often depend on your age, risk factors, and whether you use in-network providers. Eligibility and recommended frequency can vary by plan and guidelines. For detailed eligibility and frequency information, please check your plan's preventive care materials or visit healthcare.gov/preventive-care (or your insurer's preventive care page) for general guidance."
FROM_POLICY_LABEL = "From this policy document:"
GENERAL_GUIDANCE_LABEL = "General Preventive Care Guidance (Not Policy-Specific):"
PREVENTIVE_FOLLOW_UP_QUESTIONS = [
    "Is this preventive service covered in-network only?", "Are there age restrictions for this preventive service?",
    "How often can I receive this screening?", "Does it require prior authorization?",
    "Does the deductible apply to this preventive service?", "Is a copay or coinsurance required?",
    "Do I need a referral for this preventive care?", "What documentation do I need for reimbursement?",
]


def _qa_build_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"---\nChunk {c.get('chunk_id', '')} (page {c.get('page_number', 0)}):\n{(c.get('chunk_text') or '').strip()}\n")
    return "\n".join(parts).strip()


def _qa_system_prompt() -> str:
    return """You are a policy document Q&A system. Use ONLY the provided document chunks. Do not use external knowledge.
Rules:
- Answer in 1–6 sentences. Every factual claim MUST be supported by a citation from the chunks.
- Citations must use only "page" and "chunk_id" that appear in the provided chunks (e.g. {"page": 5, "chunk_id": "c_5_0"}).
- If the chunks do not contain enough information to answer, set answer_type to "not_found" and answer to "Not found in this document."
- If the question is ambiguous or lacks required specificity, set answer_type to "ambiguous", provide a clarification_question (1–2 sentences), and leave citations empty.
- If the retrieved chunks contradict each other, set answer_type to "conflict", provide a brief explanation (1–3 sentences), and cite both pages.
- Output valid JSON only. No markdown or explanation outside the JSON."""


def _qa_user_prompt(question: str, context: str) -> str:
    return f"""Question: {question}

Document excerpts (use ONLY these; cite with page and chunk_id from below):
{context}

Respond with a single JSON object with these keys:
- answer_type: one of "answerable", "not_found", "ambiguous", "conflict"
- answer: your answer text (1–6 sentences for answerable; "Not found in this document." for not_found; brief explanation for conflict; empty or N/A for ambiguous)
- citations: list of {{"page": <int>, "chunk_id": "<id>"}} — required for answerable and conflict; empty for not_found and ambiguous
- clarification_question: (only if answer_type is "ambiguous") 1–2 sentences asking for specificity
- conflict_explanation: (only if answer_type is "conflict") 1–3 sentences describing the contradiction

Output only the JSON object."""


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _validate_citations(citations: list[Any], allowed_chunk_ids: set[str]) -> list[Citation]:
    out = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        page, cid = c.get("page"), c.get("chunk_id")
        if isinstance(page, int) and page >= 1 and isinstance(cid, str) and cid and cid in allowed_chunk_ids:
            out.append(Citation(page=page, chunk_id=cid))
    return out


def ask(doc_id: str, question: str, top_k: int = DEFAULT_TOP_K) -> QAResponseOutput:
    question = (question or "").strip()
    if not question:
        return QAResponseOutput(doc_id=doc_id, question=question or "", answer=NOT_FOUND_ANSWER, answer_type="not_found", citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER, validation_issues=[], answer_display=NOT_FOUND_ANSWER)
    chunks = storage.query(doc_id, question, top_k=top_k)
    allowed_ids = {str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")}
    if not chunks or not any((c.get("chunk_text") or "").strip() for c in chunks):
        return QAResponseOutput(doc_id=doc_id, question=question, answer=NOT_FOUND_ANSWER, answer_type="not_found", citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER, validation_issues=[], answer_display=NOT_FOUND_ANSWER)
    context = _qa_build_context(chunks)
    if not context.strip():
        return QAResponseOutput(doc_id=doc_id, question=question, answer=NOT_FOUND_ANSWER, answer_type="not_found", citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER, validation_issues=[], answer_display=NOT_FOUND_ANSWER)
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(model=settings.llm_model, messages=[{"role": "system", "content": _qa_system_prompt()}, {"role": "user", "content": _qa_user_prompt(question, context)}], temperature=0.1)
    raw = (response.choices[0].message.content or "").strip()
    parsed = _parse_llm_json(raw)
    if not parsed:
        return QAResponseOutput(doc_id=doc_id, question=question, answer=NOT_FOUND_ANSWER, answer_type="not_found", citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER, validation_issues=["llm_response_parse_failed"], answer_display=NOT_FOUND_ANSWER)
    answer_type = parsed.get("answer_type") or "not_found"
    if answer_type not in ("answerable", "not_found", "ambiguous", "conflict"):
        answer_type = "not_found"
    answer_text = parsed.get("answer") or ""
    citations = _validate_citations(parsed.get("citations"), allowed_ids)
    if answer_type == "ambiguous":
        answer_text = parsed.get("clarification_question") or "Please specify what you would like to know." if not answer_text.strip() else answer_text
        citations = []
    elif answer_type == "conflict":
        if not answer_text.strip():
            answer_text = parsed.get("conflict_explanation") or "The document excerpts contain conflicting information."
    elif answer_type == "not_found":
        answer_text = NOT_FOUND_ANSWER
        citations = []
    term_map = load_terminology_map()
    answer_text = normalize_text(answer_text, term_map)
    valid_page_numbers = {c.get("page_number") for c in chunks if c.get("page_number")}
    response_json = {"answer": answer_text, "answer_type": answer_type, "citations": [{"page": c.page, "chunk_id": c.chunk_id} for c in citations], "disclaimer": QA_RESPONSE_DISCLAIMER}
    is_valid, validation_issues, answer_display = validate_qa_response(response_json, valid_page_numbers=valid_page_numbers)
    if not is_valid and "answerable_but_no_citations" in validation_issues:
        return QAResponseOutput(doc_id=doc_id, question=question, answer=NOT_FOUND_ANSWER, answer_type="not_found", citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER, validation_issues=validation_issues, answer_display=NOT_FOUND_ANSWER)
    confidence = confidence_for_qa(answer_type, len(citations), validation_issues=validation_issues, retrieval_chunk_count=len(chunks), retrieval_strong=len(chunks) >= 3)
    return QAResponseOutput(doc_id=doc_id, question=question, answer=answer_text, answer_type=answer_type, citations=citations, confidence=confidence, disclaimer=QA_RESPONSE_DISCLAIMER, validation_issues=validation_issues, answer_display=answer_display)


# --- Section deep-dive ---
def detect_section_detail_intent(question: str) -> str | None:
    if not question or not question.strip():
        return None
    q_lower = question.strip().lower()
    if not any(re.search(p, q_lower, re.IGNORECASE) for p in DETAIL_INTENT_PATTERNS):
        return None
    for section in sorted(CORE_SECTIONS, key=len, reverse=True):
        if section.lower() in q_lower:
            return section
    for section in CORE_SECTIONS:
        first_word = section.split()[0].lower()
        if len(first_word) > 3 and first_word in q_lower:
            return section
    return None


def _bullets_to_page_citations(bullets: list) -> list[PageCitation]:
    pages = set()
    for b in bullets:
        for c in b.citations:
            pages.add(c.page)
    return [PageCitation(page=p) for p in sorted(pages)]


def ask_section_detail(doc_id: str, question: str, section_name: str) -> SectionDetailQAResponseOutput:
    chunks = retrieve_for_section(doc_id, section_name)
    summary = summarize_section(section_name, chunks, detail_level="detailed")
    if not summary.present or not summary.bullets:
        return SectionDetailQAResponseOutput(doc_id=doc_id, question=question, answer_type="section_detail", section_id=section_name, answer=summary.not_found_message or NOT_FOUND_MESSAGE, bullets=[], citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER)
    page_citations = _bullets_to_page_citations(summary.bullets)
    confidence = "high" if len(summary.bullets) >= 6 else "medium"
    return SectionDetailQAResponseOutput(doc_id=doc_id, question=question, answer_type="section_detail", section_id=section_name, answer=f"Detailed summary of {section_name} ({len(summary.bullets)} bullets).", bullets=summary.bullets, citations=page_citations, confidence=confidence, disclaimer=QA_RESPONSE_DISCLAIMER)


def route_question(doc_id: str, question: str) -> SectionDetailQAResponseOutput | None:
    section_name = detect_section_detail_intent(question)
    if not section_name:
        return None
    return ask_section_detail(doc_id, question, section_name)


# --- Scenario ---
def detect_scenario_intent(question: str) -> tuple[bool, str | None]:
    if not question or not question.strip():
        return False, None
    q_lower = question.strip().lower()
    if not any(phrase in q_lower for phrase in SCENARIO_TRIGGER_PHRASES):
        return False, None
    for scenario_type, keywords in SCENARIO_TYPES.items():
        if any(kw in q_lower for kw in keywords):
            return True, scenario_type
    return True, "General"


def _retrieve_scenario_context(doc_id: str, scenario_type: str, top_k: int = 8) -> list[dict[str, Any]]:
    seen = {}
    for q in ["deductible copay coinsurance out of pocket cost", scenario_type.lower()]:
        for hit in storage.query(doc_id, q, top_k=top_k):
            cid = hit.get("chunk_id")
            if cid and cid not in seen:
                seen[cid] = hit
    return sorted(seen.values(), key=lambda x: (x.get("page_number", 0), x.get("chunk_id", "")))


def _scenario_system_prompt() -> str:
    return """You are an example scenario generator for a health policy document. Use ONLY numbers and terms from the provided chunks.
Rules: Every dollar amount, percentage, or numeric value MUST come from the chunks and MUST have a "page" citation. Output 3–6 steps. Each step: "step_number" (1-based), "text" (plain English), "citations" (list of {"page": N}). If the chunks do not contain enough information, set "not_found": true and "steps": []. Do not invent any numbers. Output valid JSON only."""


def _scenario_user_prompt(question: str, scenario_type: str, context: str) -> str:
    return f"""Question: {question}\nScenario type: {scenario_type}\n\nDocument excerpts (use ONLY these numbers; cite page for every number):\n{context}\n\nProduce JSON: "not_found": true/false, "steps": [{{"step_number": 1, "text": "...", "citations": [{{"page": N}}]}}, ...] (3–6 steps). Output only the JSON object."""


def _validate_steps(steps_raw: list[Any], allowed_pages: set[int]) -> list[ScenarioStepOutput]:
    out = []
    for i, s in enumerate(steps_raw or []):
        if not isinstance(s, dict):
            continue
        num = s.get("step_number", i + 1)
        if not isinstance(num, int) or num < 1:
            num = i + 1
        text = (s.get("text") or "").strip()
        cites = [PageCitation(page=c["page"]) for c in s.get("citations") or [] if isinstance(c, dict) and isinstance(c.get("page"), int) and c["page"] in allowed_pages]
        out.append(ScenarioStepOutput(step_number=num, text=text, citations=cites))
    return out[:6]


def ask_scenario(doc_id: str, question: str, scenario_type: str | None = None) -> ScenarioQAResponseOutput:
    if scenario_type is None:
        _, scenario_type = detect_scenario_intent(question)
        scenario_type = scenario_type or "General"
    chunks = _retrieve_scenario_context(doc_id, scenario_type)
    allowed_pages = {c.get("page_number") for c in chunks if c.get("page_number")}
    if not chunks or not any((c.get("chunk_text") or "").strip() for c in chunks):
        return ScenarioQAResponseOutput(doc_id=doc_id, question=question, answer_type="scenario", scenario_type=scenario_type, header=SCENARIO_HEADER, steps=[], not_found_message=NOT_FOUND_ANSWER, citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER)
    context = _qa_build_context(chunks)
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(model=settings.llm_model, messages=[{"role": "system", "content": _scenario_system_prompt()}, {"role": "user", "content": _scenario_user_prompt(question, scenario_type, context)}], temperature=0.1)
    raw = (response.choices[0].message.content or "").strip()
    parsed = _parse_llm_json(raw)
    if not parsed:
        return ScenarioQAResponseOutput(doc_id=doc_id, question=question, answer_type="scenario", scenario_type=scenario_type, header=SCENARIO_HEADER, steps=[], not_found_message=NOT_FOUND_ANSWER, citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER)
    not_found = bool(parsed.get("not_found"))
    steps = _validate_steps(parsed.get("steps") or [], allowed_pages)
    term_map = load_terminology_map()
    steps = [ScenarioStepOutput(step_number=s.step_number, text=normalize_text(s.text, term_map), citations=s.citations) for s in steps]
    if not_found or len(steps) < 3:
        return ScenarioQAResponseOutput(doc_id=doc_id, question=question, answer_type="scenario", scenario_type=scenario_type, header=SCENARIO_HEADER, steps=[], not_found_message=NOT_FOUND_ANSWER, citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER)
    all_pages = {c.page for s in steps for c in s.citations}
    citations = [PageCitation(page=p) for p in sorted(all_pages)]
    confidence = "high" if len(steps) >= 4 and len(citations) >= 2 else "medium"
    return ScenarioQAResponseOutput(doc_id=doc_id, question=question, answer_type="scenario", scenario_type=scenario_type, header=SCENARIO_HEADER, steps=steps, not_found_message=None, citations=citations, confidence=confidence, disclaimer=QA_RESPONSE_DISCLAIMER)


def route_scenario_question(doc_id: str, question: str) -> ScenarioQAResponseOutput | None:
    is_scenario, scenario_type = detect_scenario_intent(question)
    if not is_scenario:
        return None
    return ask_scenario(doc_id, question, scenario_type)


# --- Preventive ---
def is_preventive_question(question: str) -> bool:
    if not question or not question.strip():
        return False
    return any(kw in question.strip().lower() for kw in PREVENTIVE_KEYWORDS)


def get_preventive_follow_up_questions() -> list[str]:
    return list(PREVENTIVE_FOLLOW_UP_QUESTIONS)


def ask_preventive(doc_id: str, question: str, top_k: int = 5) -> PreventiveQAResponseOutput:
    base = ask(doc_id, question, top_k=top_k)
    add_guidance = base.answer_type in ("not_found", "ambiguous", "conflict") or (base.answer_type == "answerable" and base.confidence == "low")
    if add_guidance:
        answer_text = f"{FROM_POLICY_LABEL}\n\n{base.answer}\n\n{GENERAL_GUIDANCE_LABEL}\n\n{GENERAL_PREVENTIVE_GUIDANCE}"
    else:
        answer_text = f"{FROM_POLICY_LABEL}\n\n{base.answer}"
    return PreventiveQAResponseOutput(doc_id=base.doc_id, question=base.question, answer=answer_text, answer_type=base.answer_type, citations=base.citations, confidence=base.confidence, disclaimer=QA_RESPONSE_DISCLAIMER, follow_up_questions=get_preventive_follow_up_questions())
