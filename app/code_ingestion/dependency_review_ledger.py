from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.code_ingestion.dependency_review import DependencyReviewPacket


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DependencyReviewDecision(FrozenModel):
    review_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_canonical_name: str
    verdict: Literal["accepted", "corrected", "needs_more_context"]
    effective_dependency_kind: str
    effective_resolution_state: str
    rationale: str = Field(min_length=1)


class DependencyReviewLedger(FrozenModel):
    schema_version: Literal["code_dependency_review_ledger_v1"] = (
        "code_dependency_review_ledger_v1"
    )
    status: Literal["reviewed", "pending"]
    reviewer: str = Field(min_length=1)
    snapshot_id: str
    parser_generation: str
    analysis_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_markdown_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: tuple[DependencyReviewDecision, ...]
    external_calls_performed: bool = False

    @model_validator(mode="after")
    def validate_decisions(self) -> "DependencyReviewLedger":
        if len({decision.review_id for decision in self.decisions}) != len(self.decisions):
            raise ValueError("Dependency review decision IDs must be unique")
        expected_status = (
            "pending"
            if any(decision.verdict == "needs_more_context" for decision in self.decisions)
            else "reviewed"
        )
        if self.status != expected_status:
            raise ValueError("Ledger status does not match its decision verdicts")
        if self.ledger_identity_sha256 != _ledger_identity(self):
            raise ValueError("Dependency review ledger identity is invalid")
        return self


def import_dependency_review_markdown(
    packet_json_path: Path,
    reviewed_markdown_path: Path,
    *,
    reviewer: str,
) -> DependencyReviewLedger:
    if not reviewer.strip():
        raise ValueError("Reviewer must not be blank")
    packet_bytes = packet_json_path.read_bytes()
    markdown_bytes = reviewed_markdown_path.read_bytes()
    packet = DependencyReviewPacket.model_validate_json(packet_bytes)
    _verify_packet_identity(packet)
    markdown = markdown_bytes.decode("utf-8").replace("\r\n", "\n")
    declared_packet = _required_match(
        markdown,
        r"^- Packet SHA-256: `([0-9a-f]{64})`$",
        "packet identity",
    )
    if declared_packet != packet.packet_identity_sha256:
        raise ValueError("Reviewed Markdown packet identity does not match canonical JSON")

    sections = re.findall(
        r"^## ([0-9]+)\. (.+?)\r?\n(.*?)(?=^## [0-9]+\.|\Z)",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(sections) != len(packet.cases):
        raise ValueError("Reviewed Markdown must contain every canonical review case exactly once")

    decisions = []
    for expected_index, (case, section) in enumerate(zip(packet.cases, sections, strict=True), 1):
        ordinal, target, body = section
        if int(ordinal) != expected_index or target.strip() != case.target_canonical_name:
            raise ValueError("Reviewed Markdown case order or target differs from canonical JSON")
        review_id = _required_match(
            body,
            r"^- Review ID: `([0-9a-f]{64})`$",
            "review ID",
        )
        if review_id != case.review_id:
            raise ValueError(f"Review ID mismatch for {case.target_canonical_name}")
        verdict = _required_match(
            body,
            r"^SME verdict:[ \t]*([^\r\n]+?)[ \t]*$",
            "SME verdict",
        ).strip().lower()
        if verdict not in {"accepted", "corrected", "needs_more_context"}:
            raise ValueError(f"Invalid or placeholder SME verdict for {case.target_canonical_name}")
        corrected = _required_match(
            body,
            r"^SME corrected kind/state:[ \t]*([^\r\n]*)$",
            "corrected kind/state",
        ).strip()
        rationale = _required_match(
            body,
            r"^SME rationale:[ \t]*([^\r\n]+?)[ \t]*$",
            "SME rationale",
        ).strip()
        if verdict == "corrected":
            parts = [part.strip() for part in corrected.split("/", 1)]
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"Corrected verdict requires 'kind / state': {case.target_canonical_name}"
                )
            effective_kind, effective_state = parts
        else:
            if corrected:
                raise ValueError(
                    f"Only corrected verdicts may change kind/state: {case.target_canonical_name}"
                )
            effective_kind = case.proposed_dependency_kind
            effective_state = case.proposed_resolution_state
        decisions.append(
            DependencyReviewDecision(
                review_id=review_id,
                target_canonical_name=case.target_canonical_name,
                verdict=verdict,
                effective_dependency_kind=effective_kind,
                effective_resolution_state=effective_state,
                rationale=rationale,
            )
        )

    payload = {
        "status": (
            "pending"
            if any(decision.verdict == "needs_more_context" for decision in decisions)
            else "reviewed"
        ),
        "reviewer": reviewer.strip(),
        "snapshot_id": packet.snapshot_id,
        "parser_generation": packet.parser_generation,
        "analysis_policy_sha256": packet.analysis_policy_sha256,
        "packet_identity_sha256": packet.packet_identity_sha256,
        "packet_json_sha256": _sha256_bytes(packet_bytes),
        "reviewed_markdown_sha256": _sha256_bytes(markdown_bytes),
        "decisions": tuple(decisions),
        "external_calls_performed": False,
    }
    identity = _sha256_json(
        {
            "schema_version": "code_dependency_review_ledger_v1",
            **payload,
            "decisions": [decision.model_dump(mode="json") for decision in decisions],
        }
    )
    return DependencyReviewLedger(**payload, ledger_identity_sha256=identity)


def write_dependency_review_ledger_no_overwrite(
    ledger: DependencyReviewLedger,
    output_path: Path,
) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Dependency review ledger already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(ledger.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    observed = DependencyReviewLedger.model_validate_json(output_path.read_text(encoding="utf-8"))
    if observed != ledger:
        raise RuntimeError("Persisted dependency review ledger failed validation")
    return output_path


def load_dependency_review_ledger(path: Path) -> DependencyReviewLedger:
    return DependencyReviewLedger.model_validate_json(path.read_text(encoding="utf-8"))


def _verify_packet_identity(packet: DependencyReviewPacket) -> None:
    payload = {
        "snapshot_id": packet.snapshot_id,
        "snapshot_content_sha256": packet.snapshot_content_sha256,
        "parser_generation": packet.parser_generation,
        "analysis_policy_sha256": packet.analysis_policy_sha256,
        "cases": [case.model_dump(mode="json") for case in packet.cases],
    }
    if _sha256_json(payload) != packet.packet_identity_sha256:
        raise ValueError("Canonical dependency review packet identity is invalid")


def _ledger_identity(ledger: DependencyReviewLedger) -> str:
    payload = ledger.model_dump(mode="json", exclude={"ledger_identity_sha256"})
    return _sha256_json(payload)


def _required_match(text: str, pattern: str, field: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Reviewed Markdown is missing {field}")
    return match.group(1)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
