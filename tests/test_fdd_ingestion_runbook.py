from pathlib import Path


RUNBOOK = Path("docs/Steps_for_FDD_Ingestion.md")


def test_runbook_keeps_active_v5_out_of_ingestion_targets() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "Do not ingest new documents\ndirectly into that active pair" in text
    assert "$env:QDRANT_COLLECTION_NAME='functional_specs_v6_intake'" in text
    assert "$env:INGESTION_OUTPUT_DIR='data/staging/functional_specs_v6_intake/processed'" in text
    assert "--collection-name functional_specs_v6" in text
    assert "--stage-directory data/staging/functional_specs_v6" in text


def test_runbook_requires_paired_activation_and_evaluation() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "QDRANT_COLLECTION_NAME=functional_specs_v6" in text
    assert "PROCESSED_DIR=data/indexes/functional_specs_v6/processed" in text
    assert "run_fdd_retrieval_gate.py" in text
    assert "explicit authorization" in text
    assert "retain v5 for rollback" in text
