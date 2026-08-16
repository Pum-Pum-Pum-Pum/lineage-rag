from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.analysis_policy import load_code_analysis_policy
from app.code_ingestion.plsql_dependency_analysis import extract_dependencies
from app.code_ingestion.plsql_parser_core import parse_plsql_source
from app.code_ingestion.plsql_symbol_analysis import extract_symbols


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate labeled static dependency classification.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT_DIR / "tests/fixtures/code_dependency_classifier_v1.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT_DIR / "config/code_analysis.toml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = load_code_analysis_policy(args.policy)
    true_positive = false_positive = false_negative = 0
    boundary_total = boundary_correct = 0
    case_results = []
    for case in payload["cases"]:
        source = case["source"]
        parsed = parse_plsql_source(
            source,
            snapshot_id="dependency-eval",
            source_path=f"{case['case_id']}.sql",
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )
        symbols = extract_symbols(parsed, module_id="dependency-eval")
        edges = extract_dependencies(
            source,
            parsed,
            file_symbols=symbols,
            all_symbols=symbols,
            schema_objects=(),
            policy=policy.model_copy(
                update={
                    "boundaries": policy.boundaries.model_copy(
                        update={"kernel_package_prefixes": ("KERNEL_",)}
                    )
                }
            ),
        )
        predicted = {
            edge.target_canonical_name
            for edge in edges
            if edge.dependency_kind == "routine_call"
        }
        expected = set(case["expected_routine_calls"])
        true_positive += len(predicted & expected)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        boundary_results = {}
        for target, expected_state in case["expected_boundaries"].items():
            boundary_total += 1
            matches = [
                edge for edge in edges if edge.target_canonical_name == target
            ]
            actual = matches[0].resolution_state if len(matches) == 1 else None
            boundary_results[target] = actual
            boundary_correct += actual == expected_state
        forbidden_present = sorted(
            predicted & set(case["forbidden_routine_calls"])
        )
        case_results.append(
            {
                "case_id": case["case_id"],
                "expected_calls": sorted(expected),
                "predicted_calls": sorted(predicted),
                "false_positives": sorted(predicted - expected),
                "false_negatives": sorted(expected - predicted),
                "forbidden_present": forbidden_present,
                "boundary_results": boundary_results,
            }
        )
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    passed = (
        precision == 1.0
        and recall == 1.0
        and boundary_correct == boundary_total
        and not any(case["forbidden_present"] for case in case_results)
    )
    report = {
                "status": "pass" if passed else "fail",
                "review_status": payload["review_status"],
                "precision": precision,
                "recall": recall,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "boundary_correct": boundary_correct,
                "boundary_total": boundary_total,
                "cases": case_results,
                "external_calls_performed": False,
            }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
