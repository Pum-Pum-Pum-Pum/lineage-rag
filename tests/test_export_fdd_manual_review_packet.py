import json
from pathlib import Path

from scripts import export_fdd_manual_review_packet


def test_export_packet_includes_failed_case_and_validated_trace(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    eval_path = tmp_path / "cases.jsonl"
    trace_directory = tmp_path / "run-1" / "answer_traces"
    trace_directory.mkdir(parents=True)
    (trace_directory / "run-1-case-1.json").write_text("{}", encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "metadata": {"run_id": "run-1"},
                "cases": [
                    {
                        "case_id": "case-1",
                        "structural_passed": False,
                        "answer": "A refused answer",
                        "is_answered": False,
                        "refusal_reason": "No evidence",
                        "citation_document_ids": ["doc-1"],
                        "citation_release_labels": ["R1"],
                        "failures": ["Expected answer"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "question": "Expected question?",
                "expected_claims": ["Expected claim"],
                "required_citation_document_ids": ["doc-1"],
                "expected_release_labels": ["R1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entries = export_fdd_manual_review_packet.load_failed_review_entries(report_path, eval_path)
    output_path = export_fdd_manual_review_packet.write_manual_review_packet(
        entries, tmp_path / "packet.md"
    )

    packet = output_path.read_text(encoding="utf-8")
    assert len(entries) == 1
    assert "## case-1" in packet
    assert "SME verdict: `pending`" in packet
    assert "run-1-case-1.json" in packet
