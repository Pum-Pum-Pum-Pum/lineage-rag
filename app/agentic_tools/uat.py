from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field, model_validator

from app.agentic_tools.evaluation import FrozenModel
from app.agentic_tools.models import BoundedToolExecution


class ManualToolUatCase(FrozenModel):
    schema_version: Literal["manual_bounded_tool_uat_case_v1"] = (
        "manual_bounded_tool_uat_case_v1"
    )
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,127}$")
    source_reviewed_case_id: str
    knowledge_mode: Literal["fdd", "code", "combined"]
    question: str = Field(min_length=10, max_length=4000)
    limit: int = Field(ge=1, le=8)
    expected_outcome: Literal["evidence", "qualified_unknown"]
    expected_fdd_document_ids: tuple[str, ...] = ()
    expected_code_paths: tuple[str, ...] = ()
    expected_code_symbols: tuple[str, ...] = ()
    require_reviewed_lineage: bool = False
    review_status: Literal["draft", "reviewed"] = "draft"
    sme_reviewed: bool = False
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def validate_review_state(self) -> "ManualToolUatCase":
        if (self.review_status == "reviewed") != self.sme_reviewed:
            raise ValueError("Manual UAT review status and SME flag must agree")
        return self


class ManualToolUatCheck(FrozenModel):
    name: str
    passed: bool
    expected: tuple[str, ...] = ()
    observed: tuple[str, ...] = ()


class ManualToolUatCaseSummary(FrozenModel):
    case_id: str
    report_file: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_passed: bool
    checks: tuple[ManualToolUatCheck, ...]


class ManualToolUatBatchReport(FrozenModel):
    schema_version: Literal["manual_bounded_tool_uat_batch_v1"] = (
        "manual_bounded_tool_uat_batch_v1"
    )
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[ManualToolUatCaseSummary, ...]
    diagnostic_passes: int = Field(ge=0)
    diagnostic_total: int = Field(ge=1)
    all_cases_reviewed: Literal[False] = False
    external_api_calls: Literal[0] = 0
    batch_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LocalToolUatReport(FrozenModel):
    schema_version: Literal["local_bounded_tool_uat_v1"] = "local_bounded_tool_uat_v1"
    knowledge_mode: Literal["fdd", "code", "combined"]
    question: str
    fdd_generation: str
    code_snapshot_id: str
    lineage_artifact_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contains_internal_source_text: Literal[True] = True
    external_api_calls: Literal[0] = 0
    execution: BoundedToolExecution
    report_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_local_uat_report(
    *,
    knowledge_mode: Literal["fdd", "code", "combined"],
    question: str,
    fdd_generation: str,
    code_snapshot_id: str,
    lineage_artifact_identity_sha256: str,
    policy_sha256: str,
    execution: BoundedToolExecution,
) -> LocalToolUatReport:
    values = {
        "knowledge_mode": knowledge_mode,
        "question": question,
        "fdd_generation": fdd_generation,
        "code_snapshot_id": code_snapshot_id,
        "lineage_artifact_identity_sha256": lineage_artifact_identity_sha256,
        "policy_sha256": policy_sha256,
        "contains_internal_source_text": True,
        "external_api_calls": 0,
        "execution": execution.model_dump(mode="json"),
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return LocalToolUatReport(
        **values, report_identity_sha256=hashlib.sha256(encoded).hexdigest()
    )


def write_local_uat_report_no_overwrite(report: LocalToolUatReport, path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite local tool UAT report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(report.model_dump_json(indent=2))
    observed = LocalToolUatReport.model_validate_json(path.read_text(encoding="utf-8"))
    if observed != report:
        raise RuntimeError("Local tool UAT report failed round-trip validation")
    return path


def load_manual_uat_cases(path: Path) -> tuple[ManualToolUatCase, ...]:
    cases = tuple(
        ManualToolUatCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not cases:
        raise ValueError("Manual UAT manifest is empty")
    identities = [case.case_id for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("Manual UAT case IDs must be unique")
    return cases


def build_manual_uat_case_summary(
    *, case: ManualToolUatCase, report: LocalToolUatReport, report_file: Path
) -> ManualToolUatCaseSummary:
    fdd_ids: set[str] = set()
    code_paths: set[str] = set()
    code_symbols: set[str] = set()
    lineage_edges = 0
    unknowns: set[str] = set()
    for output in report.execution.outputs:
        if output.tool_name == "fdd_search":
            fdd_ids.update(item.document_id for item in output.evidence)
        elif output.tool_name == "code_search":
            code_paths.update(item.source_path for item in output.evidence)
            code_symbols.update(item.display_name for item in output.evidence)
        else:
            lineage_edges += sum(
                edge.edge_kind == "reviewed_implementation" for edge in output.edges
            )
            unknowns.update(output.unknowns)
    checks = [
        _coverage("fdd_documents", case.expected_fdd_document_ids, fdd_ids),
        _coverage("code_paths", case.expected_code_paths, code_paths),
        _coverage("code_symbols", case.expected_code_symbols, code_symbols),
        ManualToolUatCheck(
            name="reviewed_lineage",
            passed=not case.require_reviewed_lineage or lineage_edges > 0,
            expected=("reviewed_edge",) if case.require_reviewed_lineage else (),
            observed=("reviewed_edge",) if lineage_edges else (),
        ),
        ManualToolUatCheck(
            name="qualified_unknown",
            passed=case.expected_outcome != "qualified_unknown" or bool(unknowns),
            expected=("qualified_unknown",)
            if case.expected_outcome == "qualified_unknown"
            else (),
            observed=tuple(sorted(unknowns)),
        ),
    ]
    return ManualToolUatCaseSummary(
        case_id=case.case_id,
        report_file=str(report_file),
        report_sha256=hashlib.sha256(report.model_dump_json(indent=2).encode("utf-8")).hexdigest(),
        diagnostic_passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def build_manual_uat_batch_report(
    *, manifest_sha256: str, summaries: Sequence[ManualToolUatCaseSummary]
) -> ManualToolUatBatchReport:
    values = {
        "manifest_sha256": manifest_sha256,
        "cases": [item.model_dump(mode="json") for item in summaries],
        "diagnostic_passes": sum(item.diagnostic_passed for item in summaries),
        "diagnostic_total": len(summaries),
        "all_cases_reviewed": False,
        "external_api_calls": 0,
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ManualToolUatBatchReport(
        **values, batch_identity_sha256=hashlib.sha256(encoded).hexdigest()
    )


def write_manual_uat_packet_no_overwrite(
    *,
    cases: Sequence[ManualToolUatCase],
    batch: ManualToolUatBatchReport,
    path: Path,
) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manual UAT packet: {path}")
    summaries = {item.case_id: item for item in batch.cases}
    lines = [
        "# Bounded-tool formal manual UAT review packet",
        "",
        f"- Manifest SHA-256: `{batch.manifest_sha256}`",
        f"- Batch identity: `{batch.batch_identity_sha256}`",
        f"- Diagnostic results: **{batch.diagnostic_passes}/{batch.diagnostic_total}**",
        "- Review status: **draft**",
        "- External API calls: **0**",
        "",
        "Review the local report named for each case. Those reports contain internal",
        "source text; this packet contains identities and checks only.",
        "",
    ]
    for index, case in enumerate(cases, start=1):
        summary = summaries[case.case_id]
        lines.extend(
            [
                f"## {index}. {case.case_id}",
                "",
                f"- Source reviewed case: `{case.source_reviewed_case_id}`",
                f"- Question: {case.question}",
                f"- Mode: `{case.knowledge_mode}`",
                f"- Expected outcome: `{case.expected_outcome}`",
                f"- Diagnostic result: **{'pass' if summary.diagnostic_passed else 'fail'}**",
                f"- Local report: `{summary.report_file}`",
                f"- Local report SHA-256: `{summary.report_sha256}`",
                "",
                "### Checks",
                "",
            ]
        )
        for check in summary.checks:
            lines.append(
                f"- `{check.name}`: **{'pass' if check.passed else 'fail'}**; "
                f"expected `{list(check.expected)}`; observed `{list(check.observed)}`"
            )
        lines.extend(
            [
                "",
                "SME verdict: accepted | retrieval_gap | wrong_source | needs_more_context",
                "SME rationale:",
                "Required follow-up:",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return path


def _coverage(name: str, expected: Sequence[str], observed: set[str]) -> ManualToolUatCheck:
    normalized = {value.casefold() for value in observed}
    return ManualToolUatCheck(
        name=name,
        passed=all(value.casefold() in normalized for value in expected),
        expected=tuple(expected),
        observed=tuple(sorted(observed, key=str.casefold)),
    )
