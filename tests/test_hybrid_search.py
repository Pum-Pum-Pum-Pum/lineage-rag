from app.retrieval.hybrid_search import fuse_dense_and_lexical_results
from app.vectorstore.qdrant_search import QdrantSearchResult


def _result(unit_id: str, score: float, text: str = "evidence") -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id=unit_id,
        score=score,
        payload={
            "unit_id": unit_id,
            "release_label": "R24",
            "source_kind": "paragraph",
            "text": text,
        },
    )


def test_fuse_dense_and_lexical_results_rewards_overlap() -> None:
    dense_results = [
        _result("shared", 0.8, "Shared dense evidence"),
        _result("dense-only", 0.7, "Dense only evidence"),
    ]
    lexical_results = [
        _result("shared", 10.0, "Shared lexical evidence"),
        _result("lexical-only", 9.0, "Lexical only evidence"),
    ]

    results = fuse_dense_and_lexical_results(
        dense_results=dense_results,
        lexical_results=lexical_results,
        limit=3,
    )

    assert results[0].payload["unit_id"] == "shared"
    assert results[0].payload["contributing_retrievers"] == ["dense", "lexical"]
    assert results[0].payload["dense_rank"] == 1
    assert results[0].payload["lexical_rank"] == 1


def test_fuse_dense_and_lexical_results_respects_limit() -> None:
    results = fuse_dense_and_lexical_results(
        dense_results=[_result("dense-1", 0.8), _result("dense-2", 0.7)],
        lexical_results=[_result("lexical-1", 10.0), _result("lexical-2", 9.0)],
        limit=2,
    )

    assert len(results) == 2


def test_rrf_keeps_r24_teller_and_branch_tables_in_final_top_five() -> None:
    dense_results = [
        _result("existing-counts", 0.63),
        _result("requirements-summary", 0.59),
        _result("document-title", 0.56),
        _result("r2-background", 0.55),
        _result("traceability-matrix", 0.46),
    ]
    lexical_results = [
        _result("branch-realignment-table", 28.0),
        _result("teller-realignment-table", 23.0),
        _result("existing-counts", 20.8),
        _result("business-requirements-table", 18.0),
        _result("requirements-summary", 17.4),
        _result("r2-background", 17.0),
        _result("document-title", 11.2),
    ]

    results = fuse_dense_and_lexical_results(
        dense_results=dense_results,
        lexical_results=lexical_results,
        limit=5,
        dense_weight=0.4,
        lexical_weight=0.6,
    )

    result_ids = [result.payload["unit_id"] for result in results]
    assert "branch-realignment-table" in result_ids
    assert "teller-realignment-table" in result_ids
    assert "traceability-matrix" not in result_ids
    assert all(0.0 <= result.score <= 1.0 for result in results)
    assert results[0].payload["fusion_method"] == "weighted_rrf"
    assert "raw_rrf_score" in results[0].payload


def test_fuse_dense_and_lexical_results_rejects_invalid_inputs() -> None:
    try:
        fuse_dense_and_lexical_results([], [], limit=0)
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid limit")

    try:
        fuse_dense_and_lexical_results([], [], dense_weight=0, lexical_weight=0)
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected ValueError for zero weights")
