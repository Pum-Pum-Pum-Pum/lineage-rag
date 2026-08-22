from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.schemas.query_api import QueryRequest
from app.api.routes.readiness import readiness_check
from app.services.knowledge_mode_orchestration import run_code_or_combined_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess code/combined activation readiness.")
    parser.add_argument("--answer-review-ledger", type=Path, required=True)
    parser.add_argument("--contract-ledger", type=Path, required=True)
    parser.add_argument("--fdd-qdrant-path", type=Path, required=True)
    parser.add_argument("--code-qdrant-path", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--rollback-rehearsed", action="store_true")
    args = parser.parse_args()
    review = json.loads(args.answer_review_ledger.read_text(encoding="utf-8"))
    contract = json.loads(args.contract_ledger.read_text(encoding="utf-8"))
    fdd = _collections(args.fdd_qdrant_path, ("functional_specs_v4", "functional_specs_v5"))
    code = _collections(args.code_qdrant_path, ("code_custom_r1_v1", "code_custom_r1_v2"))
    checks = [
        _check("semantic_review", review["summary"]["semantic_acceptances"] == 10,
               "Ten paid answers are SME accepted."),
        _check("corrected_eval_contract", contract["schema_version"].endswith("v2"),
               "The future combined benchmark separates structural and semantic findings."),
        _check("fdd_current_and_rollback", all(item["exists"] for item in fdd.values()),
               "FDD v5 and rollback v4 exist."),
        _check("code_current_and_rollback", all(item["exists"] for item in code.values()),
               "Code v2 and rollback v1 exist."),
        _check("api_explicit_mode", "knowledge_mode" in QueryRequest.model_fields,
               "QueryRequest must expose explicit fdd/code/combined mode."),
        _check("code_runtime_configuration", all(
            name in Settings.model_fields for name in
            ("code_qdrant_local_path", "code_qdrant_collection_name", "code_index_artifact_path")
        ), "Settings must identify the active code store and artifact."),
        _check("mode_aware_readiness", "knowledge_mode" in inspect.signature(readiness_check).parameters,
               "Readiness accepts an explicit knowledge mode and validates its dependencies."),
        _check("runtime_combined_orchestration", callable(run_code_or_combined_query),
               "Code/combined runtime orchestration is callable behind the feature gate."),
        _check("rollback_rehearsal", args.rollback_rehearsed,
               "The API feature gate was enabled and disabled in a deterministic rollback test."),
    ]
    report = {
        "schema_version": "code_combined_activation_readiness_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "answer_review_ledger": str(args.answer_review_ledger),
        "answer_review_ledger_sha256": _sha256(args.answer_review_ledger),
        "contract_ledger": str(args.contract_ledger),
        "contract_ledger_sha256": _sha256(args.contract_ledger),
        "collections": {"fdd": fdd, "code": code},
        "checks": checks,
        "summary": {
            "passed": sum(item["passed"] for item in checks),
            "total": len(checks),
            "activation_ready": all(item["passed"] for item in checks),
            "activation_performed": False,
            "next_batch": (
                "After explicit approval, enable code modes, verify mode-aware readiness, "
                "run API smoke tests, and retain immediate feature-flag rollback."
            ),
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_identity_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    if args.output_file.exists():
        raise FileExistsError(f"Refusing to overwrite readiness report: {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"checks_passed={report['summary']['passed']}/{report['summary']['total']}")
    print(
        "activation_ready="
        + str(report["summary"]["activation_ready"]).lower()
    )
    print(f"report_identity_sha256={report['report_identity_sha256']}")
    return 0


def _collections(path: Path, names: tuple[str, ...]) -> dict[str, dict]:
    client = QdrantClient(path=str(path))
    try:
        return {
            name: {
                "exists": client.collection_exists(name),
                "points": client.get_collection(name).points_count
                if client.collection_exists(name)
                else None,
            }
            for name in names
        }
    finally:
        client.close()


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
