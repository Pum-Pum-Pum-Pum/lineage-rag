from pathlib import Path

from app.retrieval.evaluation import (
    RetrievalEvalCase,
    RetrievalEvalExpectation,
    RetrievalEvalFilters,
)
from app.retrieval.hybrid_weight_experiment import (
    build_hybrid_weight_experiment_report,
    build_weight_settings,
    parse_weight_pairs,
    write_hybrid_weight_experiment_report_to_json,
)
from app.vectorstore.qdrant_search import QdrantSearchResult


def _case(
    case_id: str,
    query: str,
    expected_text: str,
    expected_to_pass: bool = True,
) -> RetrievalEvalCase:
    return RetrievalEvalCase(
        case_id=case_id,
        query=query,
        filters=RetrievalEvalFilters(release_label="R24", source_kind="paragraph"),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=expected_to_pass,
            min_results=1,
            expected_release_label="R24",
            expected_source_kind="paragraph",
            expected_top1_contains_any=[expected_text],
            expected_text_contains_any=[expected_text],
        ),
    )


def _result(unit_id: str, score: float, text: str) -> QdrantSearchResult:
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


def test_parse_weight_pairs() -> None:
    assert parse_weight_pairs("0.8:0.2,0.5:0.5") == [(0.8, 0.2), (0.5, 0.5)]


def test_build_weight_settings_rejects_invalid_pairs() -> None:
    try:
        build_weight_settings([(0.0, 0.0)])
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected ValueError for zero weights")


def test_build_hybrid_weight_experiment_report_compares_settings() -> None:
    cases = [_case("case_1", "semantic query", "semantic evidence")]
    dense_results = {
        "case_1": [_result("dense-good", 0.9, "semantic evidence")],
    }
    lexical_results = {
        "case_1": [_result("lexical-noise", 100.0, "wrong lexical text")],
    }
    settings = build_weight_settings([(0.8, 0.2), (0.2, 0.8)])

    report = build_hybrid_weight_experiment_report(
        cases=cases,
        dense_results_by_case_id=dense_results,
        lexical_results_by_case_id=lexical_results,
        weight_settings=settings,
        limit=1,
    )

    assert report.total_settings == 2
    assert report.settings[0].expected_outcome_count == 1
    assert report.settings[1].unexpected_failure_count == 1
    assert report.best_setting_labels == ["dense_0.8_lexical_0.2"]


def test_hybrid_weight_experiment_tracks_unsafe_expected_failure_pass() -> None:
    cases = [_case("unsupported", "unsupported query", "bad marker", expected_to_pass=False)]
    dense_results = {"unsupported": []}
    lexical_results = {"unsupported": [_result("bad", 10.0, "bad marker")]}
    settings = build_weight_settings([(0.2, 0.8)])

    report = build_hybrid_weight_experiment_report(
        cases=cases,
        dense_results_by_case_id=dense_results,
        lexical_results_by_case_id=lexical_results,
        weight_settings=settings,
        limit=1,
    )

    assert report.settings[0].unsafe_expected_failure_pass_count == 1


def test_write_hybrid_weight_experiment_report_to_json(tmp_path: Path) -> None:
    cases = [_case("case_1", "semantic query", "semantic evidence")]
    dense_results = {"case_1": [_result("dense-good", 0.9, "semantic evidence")]}
    lexical_results = {"case_1": []}
    settings = build_weight_settings([(1.0, 0.0)])
    report = build_hybrid_weight_experiment_report(
        cases=cases,
        dense_results_by_case_id=dense_results,
        lexical_results_by_case_id=lexical_results,
        weight_settings=settings,
        limit=1,
    )

    output_file = write_hybrid_weight_experiment_report_to_json(
        report,
        tmp_path / "generated" / "hybrid_weight_report.json",
    )

    assert "dense_1_lexical_0" in output_file.read_text(encoding="utf-8")