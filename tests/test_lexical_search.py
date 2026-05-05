import json
from pathlib import Path

from app.retrieval.lexical_search import (
    LexicalSearchDocument,
    build_query_terms,
    load_retrieval_ready_documents,
    search_lexical_artifacts,
    search_lexical_documents,
    tokenize,
)


def _document(
    unit_id: str,
    text: str,
    release_label: str = "R24",
    source_kind: str = "paragraph",
    document_family: str = "FS_FCIS_14.7.0.0.0$ASNB",
    unit_index: int = 0,
) -> LexicalSearchDocument:
    return LexicalSearchDocument(
        document_name="example.docx",
        unit_id=unit_id,
        unit_index=unit_index,
        source_kind=source_kind,
        document_family=document_family,
        release_label=release_label,
        text=text,
    )


def test_tokenize_preserves_enterprise_identifiers() -> None:
    tokens = tokenize("B-01 report, T-1 Teller_End_of_Day, PNB/BOR")

    assert "b-01" in tokens
    assert "t-1" in tokens
    assert "teller_end_of_day" in tokens
    assert "pnb" in tokens
    assert "bor" in tokens


def test_build_query_terms_removes_common_stopwords() -> None:
    assert build_query_terms("what is the B-01 report layout") == [
        "b-01",
        "report",
        "layout",
    ]


def test_search_lexical_documents_ranks_exact_identifier_match_first() -> None:
    documents = [
        _document(
            "generic-report-unit",
            "Branch report report report summary without the exact layout identifier.",
            unit_index=0,
        ),
        _document(
            "b01-layout-unit",
            "B-01 report layout will be changed as per attached sample report.",
            unit_index=1,
        ),
    ]

    results = search_lexical_documents(
        documents,
        query_text="B-01 report layout",
        limit=2,
    )

    assert [result.payload["unit_id"] for result in results] == [
        "b01-layout-unit",
        "generic-report-unit",
    ]
    assert results[0].payload["matched_query_terms"] == ["b-01", "layout", "report"]
    assert results[0].score > results[1].score


def test_search_lexical_documents_respects_metadata_filters() -> None:
    documents = [
        _document(
            "r24-paragraph",
            "PNB means Permodalan Nasional Berhad.",
            release_label="R24",
            source_kind="paragraph",
        ),
        _document(
            "r24-table",
            "PNB | Permodalan Nasional Berhad",
            release_label="R24",
            source_kind="table",
        ),
        _document(
            "r2-table",
            "PNB table in older release",
            release_label="R2",
            source_kind="table",
            document_family="FS_FCIS_14.4.0.0.0$ASNB",
        ),
    ]

    results = search_lexical_documents(
        documents,
        query_text="PNB",
        limit=5,
        document_family="FS_FCIS_14.7.0.0.0$ASNB",
        release_label="R24",
        source_kind="table",
    )

    assert len(results) == 1
    assert results[0].payload["unit_id"] == "r24-table"
    assert results[0].payload["source_kind"] == "table"
    assert results[0].payload["release_label"] == "R24"


def test_load_retrieval_ready_documents_and_search_artifacts(tmp_path: Path) -> None:
    artifact_file = tmp_path / "example.retrieval_ready.json"
    artifact_file.write_text(
        json.dumps(
            {
                "document_name": "example.docx",
                "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
                "release_label": "R24",
                "total_units": 2,
                "units": [
                    {
                        "unit_id": "example.docx::chunk_0",
                        "unit_index": 0,
                        "source_kind": "paragraph",
                        "text": "Unrelated introduction text",
                        "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
                        "release_label": "R24",
                    },
                    {
                        "unit_id": "example.docx::chunk_1",
                        "unit_index": 1,
                        "source_kind": "paragraph",
                        "text": "B-01 Branch End of Day Report layout changed.",
                        "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
                        "release_label": "R24",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    documents = load_retrieval_ready_documents(tmp_path)
    results = search_lexical_artifacts(
        tmp_path,
        query_text="B-01 report layout",
        limit=1,
        release_label="R24",
        source_kind="paragraph",
    )

    assert len(documents) == 2
    assert len(results) == 1
    assert results[0].point_id == "example.docx::chunk_1"
    assert results[0].payload["retrieval_method"] == "lexical"


def test_search_lexical_documents_rejects_invalid_inputs() -> None:
    documents = [_document("unit-1", "Branch report evidence")]

    try:
        search_lexical_documents(documents, "branch", limit=0)
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid limit")

    try:
        search_lexical_documents(documents, "!!!", limit=1)
    except ValueError as exc:
        assert "searchable token" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty searchable query")


def test_load_retrieval_ready_documents_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    missing_directory = tmp_path / "missing"

    assert load_retrieval_ready_documents(missing_directory) == []