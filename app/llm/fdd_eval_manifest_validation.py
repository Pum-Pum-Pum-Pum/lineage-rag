from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from app.llm.fdd_grounded_evaluation import FddGroundedEvalCase


RELEASE_LABEL_IN_QUESTION_PATTERN = re.compile(r"\bR\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class FddEvalManifestIssue:
    severity: str
    code: str
    message: str
    case_id: str | None = None


@dataclass(frozen=True)
class FddEvalManifestReport:
    schema_version: str
    manifest_path: str
    manifest_sha256: str
    artifact_directory: str
    artifact_file_count: int
    indexed_document_count: int
    total_cases: int
    answered_cases: int
    abstention_cases: int
    reviewed_cases: int
    pending_review_cases: int
    release_gate_ready: bool
    errors: list[FddEvalManifestIssue]
    gate_blockers: list[FddEvalManifestIssue]
    warnings: list[FddEvalManifestIssue]


@dataclass(frozen=True)
class ArtifactCatalog:
    artifact_file_count: int
    document_releases: dict[str, frozenset[str]]


def build_artifact_catalog(artifact_directory: str | Path) -> ArtifactCatalog:
    directory = Path(artifact_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Lexical artifact directory does not exist: {directory}")

    artifact_paths = sorted(directory.glob("*.retrieval_ready.json"))
    if not artifact_paths:
        raise ValueError(f"No retrieval-ready artifacts found in: {directory}")

    document_releases: dict[str, set[str]] = {}
    for artifact_path in artifact_paths:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        units = payload.get("units")
        if not isinstance(units, list) or not units:
            raise ValueError(f"Artifact has no retrieval units: {artifact_path}")
        for unit_index, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise ValueError(
                    f"Artifact unit must be an object: {artifact_path} unit={unit_index}"
                )
            document_id = str(unit.get("document_id", "")).strip()
            release_label = str(unit.get("release_label", "")).strip()
            if not document_id or not release_label:
                raise ValueError(
                    "Artifact unit is missing document_id or release_label: "
                    f"{artifact_path} unit={unit_index}"
                )
            document_releases.setdefault(document_id, set()).add(release_label)

    return ArtifactCatalog(
        artifact_file_count=len(artifact_paths),
        document_releases={
            document_id: frozenset(releases)
            for document_id, releases in document_releases.items()
        },
    )


def validate_fdd_eval_manifest(
    *,
    cases: Sequence[FddGroundedEvalCase],
    manifest_path: str | Path,
    artifact_directory: str | Path,
) -> FddEvalManifestReport:
    path = Path(manifest_path)
    catalog = build_artifact_catalog(artifact_directory)
    errors: list[FddEvalManifestIssue] = []
    gate_blockers: list[FddEvalManifestIssue] = []
    warnings: list[FddEvalManifestIssue] = []
    questions: dict[str, str] = {}

    for case in cases:
        if RELEASE_LABEL_IN_QUESTION_PATTERN.search(case.question):
            errors.append(
                _issue(
                    "error",
                    "release_label_in_user_question",
                    "User-facing evaluation questions must not assume that users know release labels.",
                    case.case_id,
                )
            )
        normalized_question = " ".join(case.question.casefold().split())
        previous_case_id = questions.get(normalized_question)
        if previous_case_id is not None:
            errors.append(
                _issue(
                    "error",
                    "duplicate_question",
                    f"Question duplicates case {previous_case_id}.",
                    case.case_id,
                )
            )
        else:
            questions[normalized_question] = case.case_id

        for field_name in (
            "expected_claims",
            "expected_document_ids",
            "expected_release_labels",
            "required_citation_document_ids",
        ):
            values = list(getattr(case, field_name))
            _validate_string_list(
                values=values,
                field_name=field_name,
                case_id=case.case_id,
                errors=errors,
            )

        if not case.sme_reviewed:
            gate_blockers.append(
                _issue(
                    "gate_blocker",
                    "pending_sme_review",
                    f"Case is not SME-approved (review_status={case.review_status}).",
                    case.case_id,
                )
            )
        elif _is_pending_status(case.review_status):
            errors.append(
                _issue(
                    "error",
                    "inconsistent_review_status",
                    "sme_reviewed=true cannot use a pending or draft review_status.",
                    case.case_id,
                )
            )

        if case.should_abstain:
            continue

        if not case.expected_claims:
            errors.append(_issue("error", "missing_expected_claims", "Answered case has no expected claims.", case.case_id))
        if not case.expected_evidence:
            errors.append(_issue("error", "missing_expected_evidence", "Answered case has no expected evidence.", case.case_id))
        if not case.expected_document_ids:
            errors.append(_issue("error", "missing_expected_documents", "Answered case has no expected document IDs.", case.case_id))
        if not case.expected_release_labels:
            errors.append(_issue("error", "missing_expected_releases", "Answered case has no expected release labels.", case.case_id))

        unexpected_required = sorted(
            set(case.required_citation_document_ids).difference(case.expected_document_ids)
        )
        if unexpected_required:
            errors.append(
                _issue(
                    "error",
                    "required_citation_not_expected",
                    f"Required citation document IDs are absent from expected_document_ids: {unexpected_required}",
                    case.case_id,
                )
            )

        _validate_expected_evidence(case=case, errors=errors)
        _validate_catalog_identity(
            case=case,
            document_releases=catalog.document_releases,
            errors=errors,
            warnings=warnings,
        )

    reviewed_cases = sum(case.sme_reviewed for case in cases)
    return FddEvalManifestReport(
        schema_version="fdd_eval_manifest_validation_v1",
        manifest_path=str(path),
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        artifact_directory=str(Path(artifact_directory)),
        artifact_file_count=catalog.artifact_file_count,
        indexed_document_count=len(catalog.document_releases),
        total_cases=len(cases),
        answered_cases=sum(not case.should_abstain for case in cases),
        abstention_cases=sum(case.should_abstain for case in cases),
        reviewed_cases=reviewed_cases,
        pending_review_cases=len(cases) - reviewed_cases,
        release_gate_ready=not errors and not gate_blockers,
        errors=errors,
        gate_blockers=gate_blockers,
        warnings=warnings,
    )


def write_fdd_eval_manifest_report(
    report: FddEvalManifestReport,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_pending_sme_review_packet(
    cases: Sequence[FddGroundedEvalCase],
    output_path: str | Path,
) -> Path:
    pending = [case for case in cases if not case.sme_reviewed]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FDD v4 evaluation — pending SME review",
        "",
        "Automated validation does not establish domain correctness. Review each case against the named source before approval.",
        "",
    ]
    for case in pending:
        lines.extend(
            [
                f"## {case.case_id}",
                "",
                f"Question: {case.question}",
                "",
                f"Expected behavior: {'safe abstention' if case.should_abstain else 'grounded answer'}",
                "",
                "Expected claims:",
                *_bullets(case.expected_claims),
                "",
                "Expected evidence:",
                *_evidence_bullets(case.expected_evidence),
                "",
                f"Expected document IDs: {', '.join(case.expected_document_ids) or '(none)'}",
                f"Expected releases: {', '.join(case.expected_release_labels) or '(none)'}",
                f"Required citation document IDs: {', '.join(case.required_citation_document_ids) or '(none)'}",
                "",
                "SME verdict: `pending`",
                "SME rationale: `pending`",
                "Required follow-up: `pending`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validate_expected_evidence(
    *, case: FddGroundedEvalCase, errors: list[FddEvalManifestIssue]
) -> None:
    for evidence_index, evidence in enumerate(case.expected_evidence):
        if not isinstance(evidence, dict):
            errors.append(
                _issue("error", "invalid_expected_evidence", f"Evidence item {evidence_index} is not an object.", case.case_id)
            )
            continue
        document_id = str(evidence.get("document_id", "")).strip()
        evidence_text = str(evidence.get("evidence", "")).strip()
        source_kind = str(evidence.get("source_kind", "")).strip()
        if not document_id or (not evidence_text and not source_kind):
            errors.append(
                _issue(
                    "error",
                    "invalid_expected_evidence",
                    f"Evidence item {evidence_index} requires a nonblank document_id and either evidence text or source_kind.",
                    case.case_id,
                )
            )
        elif document_id not in case.expected_document_ids:
            errors.append(
                _issue(
                    "error",
                    "evidence_document_not_expected",
                    f"Evidence item {evidence_index} document_id is absent from expected_document_ids: {document_id}",
                    case.case_id,
                )
            )


def _validate_catalog_identity(
    *,
    case: FddGroundedEvalCase,
    document_releases: dict[str, frozenset[str]],
    errors: list[FddEvalManifestIssue],
    warnings: list[FddEvalManifestIssue],
) -> None:
    expected_releases = set(case.expected_release_labels)
    for document_id in case.expected_document_ids:
        indexed_releases = document_releases.get(document_id)
        if indexed_releases is None:
            errors.append(
                _issue(
                    "error",
                    "document_not_indexed",
                    f"Expected document is absent from promoted lexical artifacts: {document_id}",
                    case.case_id,
                )
            )
            continue
        if indexed_releases.isdisjoint(expected_releases):
            errors.append(
                _issue(
                    "error",
                    "document_release_mismatch",
                    f"Document {document_id} is indexed under {sorted(indexed_releases)}, not {sorted(expected_releases)}.",
                    case.case_id,
                )
            )
        if len(indexed_releases) > 1:
            warnings.append(
                _issue(
                    "warning",
                    "document_has_multiple_releases",
                    f"Document {document_id} has multiple indexed release labels: {sorted(indexed_releases)}.",
                    case.case_id,
                )
            )


def _validate_string_list(
    *,
    values: list[Any],
    field_name: str,
    case_id: str,
    errors: list[FddEvalManifestIssue],
) -> None:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            errors.append(
                _issue("error", "invalid_list_value", f"{field_name} contains a non-string or blank value.", case_id)
            )
            continue
        normalized.append(value.strip())
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        errors.append(
            _issue("error", "duplicate_list_value", f"{field_name} contains duplicates: {duplicates}", case_id)
        )


def _is_pending_status(review_status: str) -> bool:
    normalized = review_status.strip().casefold()
    return "pending" in normalized or "draft" in normalized


def _issue(severity: str, code: str, message: str, case_id: str | None = None) -> FddEvalManifestIssue:
    return FddEvalManifestIssue(
        severity=severity,
        code=code,
        message=message,
        case_id=case_id,
    )


def _bullets(values: Sequence[str]) -> list[str]:
    return [f"- {value}" for value in values] or ["- (none)"]


def _evidence_bullets(values: Sequence[dict[str, str]]) -> list[str]:
    bullets: list[str] = []
    for value in values:
        locator = ", ".join(
            str(value[key])
            for key in ("release_label", "source_kind")
            if str(value.get(key, "")).strip()
        )
        description = str(value.get("evidence", "")).strip() or "source locator"
        suffix = f" ({locator})" if locator else ""
        bullets.append(
            f"- [{value.get('document_id', '(missing document)')}] {description}{suffix}"
        )
    return bullets or ["- (none)"]
