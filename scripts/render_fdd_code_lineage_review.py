from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.fdd_code_lineage.models import FddCodeLineageArtifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an FDD/code lineage SME review packet.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Review packet already exists: {args.output}")
    artifact = FddCodeLineageArtifact.model_validate_json(
        args.artifact.read_text(encoding="utf-8")
    )
    lines = [
        "# FDD-to-code lineage SME review packet",
        "",
        f"- FDD generation: `{artifact.fdd_generation}`",
        f"- Code snapshot: `{artifact.code_snapshot_id}`",
        f"- Code artifact: `{artifact.code_artifact_identity_sha256}`",
        f"- Candidate artifact: `{artifact.artifact_identity_sha256}`",
        f"- Mappings: {len(artifact.mappings)}",
        "- Status: `candidate`",
        "",
        "Review whether each FDD is implemented by the listed visible custom files. "
        "A file-level acceptance is intentionally broad; prefer exact symbols when known.",
        "",
    ]
    for index, mapping in enumerate(artifact.mappings, start=1):
        lines.extend(
            [
                f"## {index}. {mapping.fdd_document_id}",
                "",
                f"- Release metadata: `{mapping.fdd_release_label}`",
                f"- Mapping ID: `{mapping.mapping_id}`",
                f"- Proposed rationale: {mapping.rationale}",
                "- Proposed targets:",
                "",
            ]
        )
        for target in mapping.targets:
            lines.append(
                f"  - `{target.path}` — scope `{target.selector_scope}` — {target.rationale}"
            )
        lines.extend(
            [
                "",
                "SME verdict: reviewed | rejected | needs_symbol_scope",
                "SME corrected targets/symbols:",
                "SME rationale:",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"review_packet={args.output}")


if __name__ == "__main__":
    main()
