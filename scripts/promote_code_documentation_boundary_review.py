from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.fdd_code_lineage.documentation_boundary import load_documentation_boundary_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote an accepted documentation-boundary draft into immutable reviewed evidence."
    )
    parser.add_argument("--draft-manifest", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--approval-note", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    reviewer = args.reviewer.strip()
    note = args.approval_note.strip()
    if not reviewer or not note:
        raise ValueError("Reviewer and approval note must be nonblank")
    draft_bytes = args.draft_manifest.read_bytes()
    cases = load_documentation_boundary_cases(args.draft_manifest)
    if any(case.sme_reviewed or case.review_status != "draft" for case in cases):
        raise ValueError("Only wholly draft documentation-boundary manifests may be promoted")
    reviewed_content = "".join(
        case.model_copy(
            update={"sme_reviewed": True, "review_status": "reviewed"}
        ).model_dump_json()
        + "\n"
        for case in cases
    )
    ledger = {
        "schema_version": "code_documentation_boundary_review_ledger_v1",
        "reviewer": reviewer,
        "approval_note": note,
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "draft_manifest": str(args.draft_manifest),
        "draft_manifest_sha256": hashlib.sha256(draft_bytes).hexdigest(),
        "reviewed_manifest": str(args.output),
        "reviewed_manifest_sha256": hashlib.sha256(reviewed_content.encode("utf-8")).hexdigest(),
        "cases": len(cases),
    }
    canonical = json.dumps(ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    ledger["ledger_identity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    outputs = {
        args.output: reviewed_content,
        args.ledger: json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    }
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite reviewed boundary evidence: {existing}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print("status=reviewed")
    print(f"reviewed_manifest_sha256={ledger['reviewed_manifest_sha256']}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    print("external_calls_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
