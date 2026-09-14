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

from app.code_ingestion.r3_benchmark import load_r3_benchmark_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote a reviewed R3 package-selection benchmark without overwriting history."
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
    draft = load_r3_benchmark_manifest(args.draft_manifest)
    if draft.review_status != "draft" or draft.sme_reviewed:
        raise ValueError("Only an unreviewed R3 benchmark draft may be promoted")
    reviewed = draft.model_copy(
        update={"review_status": "reviewed", "sme_reviewed": True, "reviewer": reviewer}
    )
    # Hash and write the exact same UTF-8/LF bytes.  Using ``write_text`` on
    # Windows can translate LF to CRLF after the hash has been calculated,
    # which would make the ledger bind different bytes from the reviewed file.
    reviewed_content = reviewed.model_dump_json(indent=2) + "\n"
    reviewed_bytes = reviewed_content.encode("utf-8")
    ledger = {
        "schema_version": "code_r3_benchmark_review_ledger_v1",
        "reviewer": reviewer,
        "approval_note": note,
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "draft_manifest": str(args.draft_manifest),
        "draft_manifest_sha256": hashlib.sha256(draft_bytes).hexdigest(),
        "reviewed_manifest": str(args.output),
        "reviewed_manifest_sha256": hashlib.sha256(reviewed_bytes).hexdigest(),
        "package_pairs": len(reviewed.package_pairs),
        "new_source_files": 14,
        "modified_base_source_path": reviewed.modified_base_source_path,
    }
    canonical = json.dumps(ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    ledger["ledger_identity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    outputs = {
        args.output: reviewed_bytes,
        args.ledger: (json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
    }
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite reviewed R3 benchmark evidence: {existing}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print("status=reviewed")
    print(f"reviewed_manifest_sha256={ledger['reviewed_manifest_sha256']}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    print("external_calls_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
