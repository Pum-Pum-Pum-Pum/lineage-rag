"""Bundle retained SME reviews for offline evaluation; no activation or new approval."""
from pathlib import Path
import argparse
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import load_code_index_artifact
from app.fdd_code_lineage.models import validate_lineage_artifact
from app.fdd_code_lineage.reviewed_bundle import build_bundle, validate_bundle, write_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs=3, action='append', type=Path, required=True,
                        metavar=('REVIEWED', 'CANDIDATE', 'PACKET'))
    parser.add_argument('--code-artifact', type=Path, required=True)
    parser.add_argument('--analysis-directory', type=Path, required=True)
    parser.add_argument('--fdd-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Bundle exists; use a fresh output path')
    payload = build_bundle(args.source)
    bundle = validate_bundle(payload)
    documents = {json.loads(p.read_text(encoding='utf-8'))['document_id']
                 for p in args.fdd_directory.glob('*.retrieval_ready.json')}
    summary = validate_lineage_artifact(bundle, fdd_document_ids=documents,
        code_artifact=load_code_index_artifact(args.code_artifact),
        analysis_directory=args.analysis_directory)
    write_bundle(payload, args.output)
    print(json.dumps(dict(summary, source_reviews=len(bundle.sources),
        artifact_identity_sha256=bundle.artifact_identity_sha256, output=str(args.output),
        new_sme_approvals=0, external_api_calls=0, activated=False), indent=2))


if __name__ == '__main__':
    main()
