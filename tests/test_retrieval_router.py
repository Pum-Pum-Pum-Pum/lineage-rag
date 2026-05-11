from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.retrieval.retrieval_router import route_retrieval
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


def _config(mode: str) -> RetrievalRuntimeConfig:
    return RetrievalRuntimeConfig(
        retrieval_mode=mode,
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
    )


def test_route_retrieval_uses_dense_mode_only() -> None:
    calls: list[tuple[str, int]] = []

    def dense(limit: int):
        calls.append(("dense", limit))
        return [_result("dense", 0.9)]

    def lexical(limit: int):
        calls.append(("lexical", limit))
        return [_result("lexical", 10.0)]

    routed = route_retrieval(_config("dense"), dense, lexical, limit=5)

    assert routed.retrieval_mode == "dense"
    assert routed.results[0].payload["unit_id"] == "dense"
    assert calls == [("dense", 5)]


def test_route_retrieval_uses_lexical_mode_only() -> None:
    calls: list[tuple[str, int]] = []

    def dense(limit: int):
        calls.append(("dense", limit))
        return [_result("dense", 0.9)]

    def lexical(limit: int):
        calls.append(("lexical", limit))
        return [_result("lexical", 10.0)]

    routed = route_retrieval(_config("lexical"), dense, lexical, limit=5)

    assert routed.retrieval_mode == "lexical"
    assert routed.results[0].payload["unit_id"] == "lexical"
    assert calls == [("lexical", 5)]


def test_route_retrieval_uses_hybrid_mode_with_candidate_limit_and_weights() -> None:
    calls: list[tuple[str, int]] = []

    def dense(limit: int):
        calls.append(("dense", limit))
        return [_result("shared", 0.8), _result("dense-only", 0.7)]

    def lexical(limit: int):
        calls.append(("lexical", limit))
        return [_result("shared", 10.0), _result("lexical-only", 9.0)]

    routed = route_retrieval(_config("hybrid"), dense, lexical, limit=5)

    assert routed.retrieval_mode == "hybrid"
    assert routed.results[0].payload["unit_id"] == "shared"
    assert routed.results[0].payload["retrieval_method"] == "hybrid"
    assert routed.results[0].payload["contributing_retrievers"] == ["dense", "lexical"]
    assert calls == [("dense", 10), ("lexical", 10)]


def test_route_retrieval_uses_limit_when_greater_than_candidate_limit() -> None:
    calls: list[tuple[str, int]] = []
    config = RetrievalRuntimeConfig(
        retrieval_mode="hybrid",
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=3,
    )

    def dense(limit: int):
        calls.append(("dense", limit))
        return []

    def lexical(limit: int):
        calls.append(("lexical", limit))
        return []

    route_retrieval(config, dense, lexical, limit=5)

    assert calls == [("dense", 5), ("lexical", 5)]


def test_route_retrieval_rejects_invalid_limit() -> None:
    try:
        route_retrieval(_config("dense"), lambda limit: [], lambda limit: [], limit=0)
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid retrieval limit")


def test_route_retrieval_rejects_unsupported_mode() -> None:
    try:
        route_retrieval(_config("agentic"), lambda limit: [], lambda limit: [], limit=5)
    except ValueError as exc:
        assert "Unsupported retrieval mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported retrieval mode")