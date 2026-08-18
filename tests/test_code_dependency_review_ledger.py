from __future__ import annotations

import hashlib
import json

import pytest

from app.code_ingestion.dependency_review import (
    DependencyReviewCase,
    DependencyReviewExample,
    DependencyReviewPacket,
    render_dependency_review_markdown,
)
from app.code_ingestion.dependency_review_ledger import (
    import_dependency_review_markdown,
    write_dependency_review_ledger_no_overwrite,
)


def _packet(tmp_path):
    case = DependencyReviewCase(
        review_id="b" * 64,
        target_canonical_name="PKG_MISSING_CUSTOM.GET_VALUE",
        proposed_dependency_kind="routine_call",
        proposed_resolution_state="custom_source_missing",
        confidence="high",
        review_reason="Custom source is absent.",
        occurrence_count=1,
        examples=(
            DependencyReviewExample(
                source_path="pkg_report_custom.sql",
                source_sha256="c" * 64,
                start_line=10,
                end_line=10,
                excerpt="000010: pkg_missing_custom.get_value();",
            ),
        ),
    )
    payload = {
        "snapshot_id": "fci-custom-r1-abc",
        "snapshot_content_sha256": "a" * 64,
        "parser_generation": "plsql_antlr_4_13_2_analysis_v12",
        "analysis_policy_sha256": "d" * 64,
        "cases": [case.model_dump(mode="json")],
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    packet = DependencyReviewPacket(
        snapshot_id=payload["snapshot_id"],
        snapshot_content_sha256=payload["snapshot_content_sha256"],
        parser_generation=payload["parser_generation"],
        analysis_policy_sha256=payload["analysis_policy_sha256"],
        total_review_cases=1,
        total_occurrences=1,
        packet_identity_sha256=identity,
        cases=(case,),
    )
    json_path = tmp_path / "packet.json"
    markdown_path = tmp_path / "packet.md"
    json_path.write_text(json.dumps(packet.model_dump(mode="json")), encoding="utf-8")
    markdown = render_dependency_review_markdown(packet).replace(
        "SME verdict: accepted | corrected | needs_more_context",
        "SME verdict: accepted",
    ).replace("SME rationale:", "SME rationale: Source will be supplied later.")
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def test_reviewed_markdown_import_is_hash_bound_and_no_overwrite(tmp_path) -> None:
    packet, markdown = _packet(tmp_path)
    ledger = import_dependency_review_markdown(packet, markdown, reviewer="project-sme")

    assert ledger.status == "reviewed"
    assert ledger.decisions[0].effective_resolution_state == "custom_source_missing"
    assert ledger.packet_json_sha256 == hashlib.sha256(packet.read_bytes()).hexdigest()
    output = write_dependency_review_ledger_no_overwrite(ledger, tmp_path / "ledger.json")
    with pytest.raises(FileExistsError):
        write_dependency_review_ledger_no_overwrite(ledger, output)


def test_placeholder_or_tampered_review_fails_closed(tmp_path) -> None:
    packet, markdown = _packet(tmp_path)
    markdown.write_text(
        markdown.read_text(encoding="utf-8").replace(
            "SME verdict: accepted",
            "SME verdict: accepted | corrected | needs_more_context",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="placeholder"):
        import_dependency_review_markdown(packet, markdown, reviewer="project-sme")

    payload = json.loads(packet.read_text(encoding="utf-8"))
    payload["cases"][0]["target_canonical_name"] = "TAMPERED"
    packet.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="packet identity"):
        import_dependency_review_markdown(packet, markdown, reviewer="project-sme")
