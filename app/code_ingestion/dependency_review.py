from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.code_ingestion.code_analysis_models import (
    CodeStaticAnalysisArtifact,
    DependencyEdge,
)
from app.code_ingestion.plsql_models import CodeParseStageManifest
from app.code_ingestion.snapshot_builder import load_snapshot_manifest


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DependencyReviewExample(FrozenModel):
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source_symbol_occurrence_id: str | None = None
    excerpt: str


class DependencyReviewCase(FrozenModel):
    review_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_canonical_name: str
    proposed_dependency_kind: str
    proposed_resolution_state: str
    confidence: str
    review_reason: str
    occurrence_count: int = Field(gt=0)
    examples: tuple[DependencyReviewExample, ...]
    sme_verdict: None = None
    sme_rationale: None = None


class DependencyReviewPacket(FrozenModel):
    schema_version: str = "code_dependency_review_packet_v1"
    review_status: str = "draft"
    snapshot_id: str
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_generation: str
    analysis_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_review_cases: int = Field(ge=0)
    total_occurrences: int = Field(ge=0)
    packet_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[DependencyReviewCase, ...]
    external_calls_performed: bool = False


def build_dependency_review_packet(
    snapshot_directory: Path,
    parse_stage_directory: Path,
    *,
    examples_per_case: int = 3,
) -> DependencyReviewPacket:
    if examples_per_case <= 0:
        raise ValueError("examples_per_case must be greater than zero")
    snapshot = load_snapshot_manifest(snapshot_directory, verify_sources=True)
    manifest = CodeParseStageManifest.model_validate_json(
        (parse_stage_directory / "parse_stage_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.snapshot_id != snapshot.snapshot_id:
        raise ValueError("Snapshot and parse-stage identities do not match")
    if manifest.snapshot_content_sha256 != snapshot.snapshot_content_sha256:
        raise ValueError("Snapshot content hash does not match the parse stage")
    if manifest.status == "failed":
        raise ValueError("A failed parse stage cannot produce an SME review packet")

    entries = {entry.path: entry for entry in snapshot.files}
    grouped: dict[tuple[str, str, str, str], list[DependencyEdge]] = defaultdict(list)
    for relative_path in manifest.analysis_artifacts:
        artifact = CodeStaticAnalysisArtifact.model_validate_json(
            (parse_stage_directory / relative_path).read_text(encoding="utf-8")
        )
        if artifact.analysis_policy_sha256 != manifest.analysis_policy_sha256:
            raise ValueError(f"Analysis policy hash mismatch: {artifact.source_path}")
        for edge in artifact.dependencies:
            if _requires_review(edge):
                key = (
                    edge.target_canonical_name,
                    edge.dependency_kind,
                    edge.resolution_state,
                    edge.confidence,
                )
                grouped[key].append(edge)

    cases: list[DependencyReviewCase] = []
    for key in sorted(grouped):
        target, kind, state, confidence = key
        edges = sorted(
            grouped[key],
            key=lambda item: (
                item.source_path.casefold(),
                item.source_map.start_offset,
                item.edge_id,
            ),
        )
        examples = []
        for edge in edges[:examples_per_case]:
            entry = entries[edge.source_path]
            source_path = snapshot_directory / snapshot.source_directory_name / entry.path
            source_text = source_path.read_bytes().decode(entry.encoding)
            examples.append(
                DependencyReviewExample(
                    source_path=edge.source_path,
                    source_sha256=entry.sha256,
                    start_line=edge.source_map.start_line,
                    end_line=edge.source_map.end_line,
                    source_symbol_occurrence_id=edge.source_symbol_occurrence_id,
                    excerpt=_line_excerpt(source_text, edge.source_map.start_line, edge.source_map.end_line),
                )
            )
        review_id = _sha256(
            json.dumps(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "policy": manifest.analysis_policy_sha256,
                    "target": target,
                    "kind": kind,
                    "state": state,
                    "confidence": confidence,
                    "edge_ids": [edge.edge_id for edge in edges],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        cases.append(
            DependencyReviewCase(
                review_id=review_id,
                target_canonical_name=target,
                proposed_dependency_kind=kind,
                proposed_resolution_state=state,
                confidence=confidence,
                review_reason=_review_reason(kind, state, confidence),
                occurrence_count=len(edges),
                examples=tuple(examples),
            )
        )

    identity_payload = {
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_content_sha256": snapshot.snapshot_content_sha256,
        "parser_generation": manifest.parser_generation,
        "analysis_policy_sha256": manifest.analysis_policy_sha256,
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    return DependencyReviewPacket(
        snapshot_id=snapshot.snapshot_id,
        snapshot_content_sha256=snapshot.snapshot_content_sha256,
        parser_generation=manifest.parser_generation,
        analysis_policy_sha256=manifest.analysis_policy_sha256,
        total_review_cases=len(cases),
        total_occurrences=sum(case.occurrence_count for case in cases),
        packet_identity_sha256=_sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
        ),
        cases=tuple(cases),
    )


def render_dependency_review_markdown(packet: DependencyReviewPacket) -> str:
    lines = [
        "# Custom-code dependency SME review packet",
        "",
        f"- Snapshot: `{packet.snapshot_id}`",
        f"- Parser generation: `{packet.parser_generation}`",
        f"- Policy SHA-256: `{packet.analysis_policy_sha256}`",
        f"- Packet SHA-256: `{packet.packet_identity_sha256}`",
        f"- Review cases: {packet.total_review_cases}",
        f"- Source occurrences represented: {packet.total_occurrences}",
        "- Review status: `draft`",
        "",
        "For each case, confirm or correct the dependency kind and resolution state.",
        "Do not infer runtime behavior beyond the displayed static evidence.",
        "",
    ]
    for index, case in enumerate(packet.cases, start=1):
        lines.extend(
            [
                f"## {index}. {case.target_canonical_name}",
                "",
                f"- Review ID: `{case.review_id}`",
                f"- Proposed kind: `{case.proposed_dependency_kind}`",
                f"- Proposed state: `{case.proposed_resolution_state}`",
                f"- Confidence: `{case.confidence}`",
                f"- Reason: {case.review_reason}",
                f"- Occurrences: {case.occurrence_count}",
                "",
            ]
        )
        for example in case.examples:
            lines.extend(
                [
                    f"### Evidence: `{example.source_path}:{example.start_line}`",
                    "",
                    "```sql",
                    example.excerpt,
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "SME verdict: accepted | corrected | needs_more_context",
                "SME corrected kind/state: ",
                "SME rationale: ",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _requires_review(edge: DependencyEdge) -> bool:
    if edge.dependency_kind == "dynamic_sql":
        return True
    if edge.dependency_kind == "routine_call" and edge.resolution_state == "ambiguous":
        return True
    return edge.dependency_kind == "kernel_boundary" and edge.confidence != "high"


def _review_reason(kind: str, state: str, confidence: str) -> str:
    if kind == "dynamic_sql":
        return "The runtime SQL target is not statically provable."
    if kind == "kernel_boundary" and confidence != "high":
        return "Kernel classification was inferred from the approved naming convention."
    if state == "ambiguous":
        return "More than one static target remains plausible."
    if state == "custom_source_missing":
        return "The target follows the custom naming contract but its source is absent from this snapshot."
    return "The static analyzer could not resolve the call in the approved snapshot."


def _line_excerpt(source_text: str, start_line: int, end_line: int) -> str:
    lines = source_text.splitlines()
    first = max(1, start_line - 2)
    last = min(len(lines), end_line + 2)
    return "\n".join(f"{line_number:06d}: {lines[line_number - 1]}" for line_number in range(first, last + 1))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
