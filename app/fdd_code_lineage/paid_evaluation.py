from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from openai import OpenAI

from app.code_retrieval.answer_contract import (
    CodeAnswerResponse,
    CodeUnknownBoundary,
    finalize_code_answer,
)
from app.code_retrieval.models import CodeRetrievalResult
from app.fdd_code_lineage.combined_answer import (
    CombinedAnswerDraft,
    CombinedAnswerResponse,
    finalize_combined_answer,
)
from app.fdd_code_lineage.combined_retrieval import CombinedRetrievalResult
from app.fdd_code_lineage.evaluation import CodeCombinedEvalCase


CODE_SYSTEM_PROMPT = """You are a grounded custom PL/SQL analysis assistant.
Use only the supplied original code excerpts. Do not use outside knowledge.
Treat visible code as the implementation source of truth for what it actually does.
Do not infer unavailable Java/kernel behavior, runtime dynamic SQL targets, external
schema behavior, or a proven root cause. Do not generate a patch.
Start with exactly DECISION: ANSWER or DECISION: REFUSE.
Every material answered claim must cite one or more supplied [C#] identifiers.
If the evidence does not directly support the request, refuse safely and explain
what source would be needed. Impact locations are candidates, not proven causes."""


COMBINED_SYSTEM_PROMPT = """You are a grounded FDD and custom PL/SQL analysis assistant.
Use only the supplied original evidence excerpts. Do not use outside knowledge.
Visible code is authoritative for what the supplied custom implementation actually
does; FDD evidence is authoritative for documented requirements and intent. Keep
those claims in separate sections. Do not infer hidden Java/kernel behavior,
runtime dynamic SQL targets, external schemas, or a proven root cause. Never
generate a patch.
Return one JSON object with exactly these keys:
requested_claim_supported, documented_functionality, visible_custom_implementation,
impact_and_likely_change_locations, unknown_or_unavailable_behavior.
requested_claim_supported must be false when the user's exact requested fact
cannot be established, even if separately labelled related context is useful.
Each value must be an object with status (answered or refused) and text.
Documented functionality may cite only [F#]. Implementation and impact may cite
only [C#]. Every material answered claim needs a valid citation. Refuse each
unsupported section independently. Impact locations are candidates, not proven
causes."""


@dataclass(frozen=True)
class PaidCall:
    request_id: str | None
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def create_no_retry_client(*, api_key: str, base_url: str | None) -> OpenAI:
    if not api_key.strip():
        raise ValueError("OPENAI_API_KEY is required for paid evaluation")
    return OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        max_retries=0,
        timeout=120.0,
    )


def embed_one_query(
    *, client: Any, model: str, question: str, expected_dimension: int
) -> tuple[list[float], dict[str, Any]]:
    response = client.embeddings.create(model=model, input=[question.strip()])
    if len(response.data) != 1 or int(response.data[0].index) != 0:
        raise RuntimeError("Embedding response did not contain exactly index 0")
    vector = [float(value) for value in response.data[0].embedding]
    if len(vector) != expected_dimension:
        raise RuntimeError(
            f"Query vector dimension mismatch: {len(vector)} != {expected_dimension}"
        )
    usage = getattr(response, "usage", None)
    return vector, {
        "request_id": getattr(response, "_request_id", None),
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def generate_grounded_answer(
    *,
    client: Any,
    model: str,
    case: CodeCombinedEvalCase,
    retrieval: CodeRetrievalResult | CombinedRetrievalResult,
    conversation_context: str | None = None,
) -> tuple[CodeAnswerResponse | CombinedAnswerResponse, dict[str, Any]]:
    system_prompt, user_prompt = build_paid_prompt(
        case=case,
        retrieval=retrieval,
        conversation_context=conversation_context,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if len(response.choices) != 1:
        raise RuntimeError("Answer response did not contain exactly one choice")
    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        raise RuntimeError("Answer response content was empty")
    usage = getattr(response, "usage", None)
    call = PaidCall(
        request_id=getattr(response, "_request_id", None),
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
    if isinstance(retrieval, CodeRetrievalResult):
        unknowns = tuple(
            CodeUnknownBoundary(kind=kind, detail=_unknown_detail(kind))
            for kind in case.expected_unknown_kinds
        )
        answer = finalize_code_answer(
            query=case.question,
            generated_content=raw,
            evidence=retrieval.evidence,
            analysis_kind=case.analysis_kind,
            additional_unknowns=unknowns,
        )
    else:
        answer = finalize_combined_answer(
            retrieval=retrieval,
            draft=CombinedAnswerDraft.model_validate(_parse_json_object(raw)),
        )
    return answer, {
        **asdict(call),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": raw.strip(),
    }


def build_paid_prompt(
    *,
    case: CodeCombinedEvalCase,
    retrieval: CodeRetrievalResult | CombinedRetrievalResult,
    conversation_context: str | None = None,
) -> tuple[str, str]:
    context = (
        "Conversation context for reference resolution only; it is not source "
        "evidence and must not support or receive citations:\n"
        f"{conversation_context.strip()}\n\n"
        if conversation_context and conversation_context.strip()
        else ""
    )
    if isinstance(retrieval, CodeRetrievalResult):
        evidence = "\n\n".join(
            _code_block(index, item)
            for index, item in enumerate(retrieval.evidence, start=1)
        ) or "No code evidence was retrieved."
        task = (
            f"Analysis kind: {case.analysis_kind}\nQuestion: {case.question}\n\n{context}"
            f"Original code evidence:\n{evidence}"
        )
        return CODE_SYSTEM_PROMPT, task

    fdd = "\n\n".join(
        "\n".join(
            (
                f"[F{index}]",
                f"document_id: {item.document_id}",
                f"release_label: {item.release_label}",
                f"source_kind: {item.source_kind}",
                f"text:\n{item.text}",
            )
        )
        for index, item in enumerate(retrieval.fdd_evidence, start=1)
    ) or "No FDD evidence was retrieved."
    code = "\n\n".join(
        _code_block(index, item)
        for index, item in enumerate(retrieval.code_evidence, start=1)
    ) or "No code evidence was retrieved."
    task = (
        f"Analysis kind: {case.analysis_kind}\nQuestion: {case.question}\n\n{context}"
        f"Original FDD evidence:\n{fdd}\n\nOriginal code evidence:\n{code}\n\n"
        "Retrieval boundary notes:\n"
        + ("\n".join(f"- {item}" for item in retrieval.unknowns) or "- None recorded")
    )
    return COMBINED_SYSTEM_PROMPT, task


def evaluate_answer_structure(
    *,
    case: CodeCombinedEvalCase,
    answer: CodeAnswerResponse | CombinedAnswerResponse,
) -> dict[str, Any]:
    failures: list[str] = []
    cited_paths: set[str]
    if isinstance(answer, CodeAnswerResponse):
        answered = answer.is_answered
        cited_paths = {item.source_path for item in answer.citations}
        cited_symbols = {item.display_name for item in answer.citations}
        if case.should_abstain and answered:
            failures.append("Expected safe refusal but the answer was marked answered")
        if not case.should_abstain and not answered:
            failures.append("Expected an answered code response but received refusal")
    else:
        substantive = (
            answer.documented_functionality,
            answer.visible_custom_implementation,
            answer.impact_and_likely_change_locations,
        )
        answered = any(item.status == "answered" for item in substantive)
        cited_paths = {item.source_path for item in answer.code_citations}
        cited_symbols = {item.display_name for item in answer.code_citations}
        if case.should_abstain and answer.requested_claim_supported:
            failures.append("Expected the requested claim to be marked unsupported")
        if not case.should_abstain:
            if not answer.requested_claim_supported:
                failures.append("Expected the requested claim to be marked supported")
            if answer.documented_functionality.status != "answered":
                failures.append("Documented functionality did not answer")
            if answer.visible_custom_implementation.status != "answered":
                failures.append("Visible custom implementation did not answer")
        cited_documents = {item.document_id for item in answer.fdd_citations}
        missing_documents = sorted(set(case.expected_fdd_document_ids) - cited_documents)
        if missing_documents:
            failures.append(f"Missing expected cited FDD documents: {missing_documents}")

    missing_paths = sorted(set(case.expected_code_paths) - cited_paths)
    missing_symbols = sorted(set(case.expected_code_symbols) - cited_symbols)
    if missing_paths:
        failures.append(f"Missing expected cited code paths: {missing_paths}")
    if missing_symbols and case.expected_code_symbol_policy == "all":
        failures.append(f"Missing expected cited code symbols: {missing_symbols}")
    if (
        case.expected_code_symbols
        and case.expected_code_symbol_policy == "any"
        and not set(case.expected_code_symbols).intersection(cited_symbols)
    ):
        failures.append(
            "No alternative expected code symbol was cited: "
            f"{list(case.expected_code_symbols)}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "semantic_sme_review_required": True,
    }


def _code_block(index: int, item: Any) -> str:
    return "\n".join(
        (
            f"[C{index}]",
            f"snapshot_id: {item.snapshot_id}",
            f"path: {item.source_path}",
            f"symbol: {item.display_name}",
            f"lines: {item.start_line}-{item.end_line}",
            f"parser_state: {item.parser_state}",
            f"conditional_state: {item.conditional_state}",
            f"text:\n{item.text}",
        )
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Combined answer must be a JSON object")
    return value


def _unknown_detail(kind: str) -> str:
    details = {
        "kernel_unavailable": "Hidden kernel implementation is absent from the approved snapshot.",
        "dynamic_sql_unknown": "The runtime dynamic SQL target cannot be proven statically.",
        "external_schema": "The external schema target is outside the approved snapshot.",
        "conditional_unknown": "The deployed conditional branch cannot be confirmed.",
        "parser_degradation": "The relevant source was parsed with reduced confidence.",
        "missing_snapshot": "The required approved source snapshot is unavailable.",
    }
    return details[kind]
