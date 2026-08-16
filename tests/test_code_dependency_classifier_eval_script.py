from __future__ import annotations

import json

import pytest

from scripts import evaluate_code_dependency_classifier


def test_labeled_dependency_fixture_reports_precision_recall_and_boundaries(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        evaluate_code_dependency_classifier.main([])

    assert error.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "pass"
    assert report["review_status"] == "draft"
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["boundary_correct"] == report["boundary_total"] == 4
    assert report["external_calls_performed"] is False


def test_wrong_expected_call_fails_closed(tmp_path, capsys) -> None:
    manifest = tmp_path / "wrong.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "code_dependency_classifier_eval_v1",
                "review_status": "draft",
                "cases": [
                    {
                        "case_id": "wrong",
                        "source": "CREATE OR REPLACE PROCEDURE p IS BEGIN NULL; END; /",
                        "expected_routine_calls": ["MISSING_CALL"],
                        "expected_boundaries": {},
                        "forbidden_routine_calls": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        evaluate_code_dependency_classifier.main(["--manifest", str(manifest)])

    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "fail"
    assert report["recall"] == 0.0
