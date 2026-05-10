from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from app.retrieval.evaluation import RetrievalEvalCase
from app.retrieval.hybrid_evaluation import (
    HybridRetrievalEvalCaseReport,
    build_hybrid_retrieval_eval_case_report,
)
from app.retrieval.hybrid_search import fuse_dense_and_lexical_results


DEFAULT_WEIGHT_PAIRS = [
    (0.8, 0.2),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.2, 0.8),
]


@dataclass(frozen=True)
class HybridWeightSetting:
    dense_weight: float
    lexical_weight: float
    label: str


@dataclass(frozen=True)
class HybridWeightSettingReport:
    setting: HybridWeightSetting
    total_cases: int
    hybrid_passed_count: int
    expected_outcome_count: int
    unsafe_expected_failure_pass_count: int
    unexpected_failure_count: int
    outcome_counts: dict[str, int]
    cases: list[HybridRetrievalEvalCaseReport]


@dataclass(frozen=True)
class HybridWeightExperimentReport:
    total_settings: int
    best_setting_labels: list[str]
    settings: list[HybridWeightSettingReport]


def build_weight_settings(
    weight_pairs: Sequence[tuple[float, float]] = DEFAULT_WEIGHT_PAIRS,
) -> list[HybridWeightSetting]:
    """Build validated hybrid weight settings from dense/lexical pairs."""

    settings: list[HybridWeightSetting] = []
    for dense_weight, lexical_weight in weight_pairs:
        if dense_weight < 0 or lexical_weight < 0:
            raise ValueError("Hybrid experiment weights must be non-negative")
        if dense_weight == 0 and lexical_weight == 0:
            raise ValueError("At least one hybrid experiment weight must be greater than 0")
        settings.append(
            HybridWeightSetting(
                dense_weight=dense_weight,
                lexical_weight=lexical_weight,
                label=format_weight_label(dense_weight, lexical_weight),
            )
        )
    return settings


def parse_weight_pairs(raw_weights: str) -> list[tuple[float, float]]:
    """Parse weight pairs like `0.8:0.2,0.5:0.5`."""

    pairs: list[tuple[float, float]] = []
    for raw_pair in raw_weights.split(","):
        stripped_pair = raw_pair.strip()
        if not stripped_pair:
            continue
        try:
            dense_raw, lexical_raw = stripped_pair.split(":", maxsplit=1)
            pairs.append((float(dense_raw), float(lexical_raw)))
        except ValueError as exc:
            raise ValueError(
                "Weight pairs must use format 'dense:lexical', for example '0.8:0.2,0.5:0.5'"
            ) from exc

    if not pairs:
        raise ValueError("At least one hybrid weight pair must be provided")
    return pairs


def build_hybrid_weight_experiment_report(
    cases: Sequence[RetrievalEvalCase],
    dense_results_by_case_id: dict[str, Sequence[Any]],
    lexical_results_by_case_id: dict[str, Sequence[Any]],
    weight_settings: Sequence[HybridWeightSetting],
    limit: int = 5,
) -> HybridWeightExperimentReport:
    """Evaluate multiple hybrid weight settings over precomputed dense/lexical candidates."""

    if limit <= 0:
        raise ValueError("Hybrid experiment limit must be greater than 0")
    if not weight_settings:
        raise ValueError("At least one hybrid weight setting is required")

    setting_reports = [
        build_hybrid_weight_setting_report(
            cases=cases,
            dense_results_by_case_id=dense_results_by_case_id,
            lexical_results_by_case_id=lexical_results_by_case_id,
            weight_setting=setting,
            limit=limit,
        )
        for setting in weight_settings
    ]
    best_setting_labels = select_best_weight_setting_labels(setting_reports)

    return HybridWeightExperimentReport(
        total_settings=len(setting_reports),
        best_setting_labels=best_setting_labels,
        settings=setting_reports,
    )


def build_hybrid_weight_setting_report(
    cases: Sequence[RetrievalEvalCase],
    dense_results_by_case_id: dict[str, Sequence[Any]],
    lexical_results_by_case_id: dict[str, Sequence[Any]],
    weight_setting: HybridWeightSetting,
    limit: int = 5,
) -> HybridWeightSettingReport:
    """Evaluate one hybrid weight setting over all cases."""

    case_reports: list[HybridRetrievalEvalCaseReport] = []
    for case in cases:
        dense_results = dense_results_by_case_id.get(case.case_id, [])
        lexical_results = lexical_results_by_case_id.get(case.case_id, [])
        hybrid_results = fuse_dense_and_lexical_results(
            dense_results=dense_results,
            lexical_results=lexical_results,
            limit=limit,
            dense_weight=weight_setting.dense_weight,
            lexical_weight=weight_setting.lexical_weight,
        )
        case_reports.append(
            build_hybrid_retrieval_eval_case_report(
                case=case,
                dense_results=list(dense_results)[:limit],
                lexical_results=list(lexical_results)[:limit],
                hybrid_results=hybrid_results,
            )
        )

    outcome_counts = Counter(report.hybrid_outcome for report in case_reports)
    return HybridWeightSettingReport(
        setting=weight_setting,
        total_cases=len(case_reports),
        hybrid_passed_count=sum(int(report.hybrid_evaluation.passed) for report in case_reports),
        expected_outcome_count=sum(int(report.hybrid_evaluation.outcome_as_expected) for report in case_reports),
        unsafe_expected_failure_pass_count=sum(
            int(
                not report.case.expectation.expected_to_pass
                and report.hybrid_evaluation.passed
            )
            for report in case_reports
        ),
        unexpected_failure_count=sum(
            int(
                report.case.expectation.expected_to_pass
                and not report.hybrid_evaluation.passed
            )
            for report in case_reports
        ),
        outcome_counts=dict(sorted(outcome_counts.items())),
        cases=case_reports,
    )


def select_best_weight_setting_labels(
    setting_reports: Sequence[HybridWeightSettingReport],
) -> list[str]:
    """Select best weight settings with safety first, then expected outcomes."""

    if not setting_reports:
        return []

    ranked = sorted(
        setting_reports,
        key=lambda report: (
            report.unsafe_expected_failure_pass_count,
            -report.expected_outcome_count,
            report.unexpected_failure_count,
            -report.hybrid_passed_count,
            report.setting.label,
        ),
    )
    best = ranked[0]
    return [
        report.setting.label
        for report in ranked
        if (
            report.unsafe_expected_failure_pass_count == best.unsafe_expected_failure_pass_count
            and report.expected_outcome_count == best.expected_outcome_count
            and report.unexpected_failure_count == best.unexpected_failure_count
            and report.hybrid_passed_count == best.hybrid_passed_count
        )
    ]


def write_hybrid_weight_experiment_report_to_json(
    report: HybridWeightExperimentReport,
    output_path: str | Path,
) -> Path:
    """Persist hybrid weight experiment results as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def format_weight_label(dense_weight: float, lexical_weight: float) -> str:
    return f"dense_{dense_weight:g}_lexical_{lexical_weight:g}"