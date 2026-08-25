from __future__ import annotations

import hashlib
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
    build_combined_contract_refusal,
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
Every citation must use exact square-bracket syntax such as [C1] or [C2]. Never
write a bare citation such as C1, "Evidence: C1", or "Evidence: C2".
If the evidence does not directly support the request, refuse safely and explain
what source would be needed. Impact locations are candidates, not proven causes."""


COMBINED_SYSTEM_PROMPT = """You are a grounded FDD and custom PL/SQL analysis assistant.
Use only the supplied original evidence excerpts. Do not use outside knowledge.
Visible code is authoritative for what the supplied custom implementation actually
does; FDD evidence is authoritative for documented requirements and intent. Keep
those claims in separate sections. Do not infer hidden Java/kernel behavior,
runtime dynamic SQL targets, external schemas, or a proven root cause. Never
generate a patch.
The response is constrained by the supplied JSON schema.
requested_claim_supported is one JSON boolean, never a section object. Set it to
false when the user's exact requested fact cannot be established, even if
separately labelled related context is useful. The other four fields are section
objects with status (answered or refused) and text.
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
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response_format = None
    response_schema_sha256 = None
    if isinstance(retrieval, CombinedRetrievalResult):
        response_format = combined_response_format()
        request["response_format"] = response_format
        response_schema_sha256 = _canonical_sha256(response_format)
    response = client.chat.completions.create(
        **request,
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
    contract_valid = True
    contract_error: str | None = None
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
        try:
            answer = finalize_combined_answer(
                retrieval=retrieval,
                draft=CombinedAnswerDraft.model_validate(_parse_json_object(raw)),
            )
        except ValueError as exc:
            contract_valid = False
            contract_error = type(exc).__name__
            answer = build_combined_contract_refusal(retrieval=retrieval)
    return answer, {
        **asdict(call),
        "contract_valid": contract_valid,
        "contract_error": contract_error,
        "response_format": "json_schema" if response_format else "text",
        "response_schema_sha256": response_schema_sha256,
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


def combined_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "combined_grounded_answer",
            "strict": True,
            "schema": CombinedAnswerDraft.model_json_schema(),
        },
    }


def combined_response_contract_aligned() -> bool:
    """Return whether the prompt and enforced schema agree on the support field."""
    response_format = combined_response_format()
    schema = response_format["json_schema"]["schema"]
    support_schema = schema.get("properties", {}).get("requested_claim_supported", {})
    return (
        response_format["type"] == "json_schema"
        and response_format["json_schema"]["strict"] is True
        and support_schema.get("type") == "boolean"
        and "requested_claim_supported is one JSON boolean" in COMBINED_SYSTEM_PROMPT
        and "Each value must be an object" not in COMBINED_SYSTEM_PROMPT
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
