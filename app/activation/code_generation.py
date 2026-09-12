"""Approval-bound local generation promotion; never starts services or paid calls."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from app.activation.code_modes import ActivationApproval, approval_identity, _atomic_write
from app.code_indexing.contract import load_code_index_artifact, build_code_index_artifact
from app.code_indexing.qdrant import verify_code_collection
from app.code_ingestion.dependency_review_ledger import DependencyReviewLedger, _ledger_identity
from app.fdd_code_lineage.models import validate_lineage_artifact
from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
from app.retrieval.lexical_search import load_retrieval_ready_documents
from app.core.config import Settings

KEYS = ('CODE_MODES_ENABLED', 'CODE_INDEX_ARTIFACT_PATH', 'CODE_ANALYSIS_DIRECTORY',
        'CODE_QDRANT_COLLECTION_NAME', 'FDD_CODE_LINEAGE_ARTIFACT_PATH')
SCHEMA = 'code_generation_promotion_request_v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def settings_identity(settings) -> str:
    # Never retain credentials or dump Settings into an audit artifact.
    return digest(settings.model_dump(mode='json', exclude={'openai_api_key', 'audit_hmac_key'}))


def local(root: Path, value) -> Path:
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Promotion inputs must stay inside the project')
    return path


def file_set(root: Path, paths) -> dict[str, str]:
    return {str(local(root, p).relative_to(root).as_posix()): sha(local(root, p)) for p in sorted(set(paths))}


def runtime_files(root: Path) -> dict[str, str]:
    paths = list((root/'app').rglob('*.py'))
    paths += list((root/'scripts').glob('*.py')) + list((root/'scripts').glob('*.ps1'))
    paths += list((root/'config').glob('*.toml'))
    paths += [root/'pyproject.toml', root/'uv.lock']
    return file_set(root, paths)


def env_values(text: str) -> dict[str, str | None]:
    result = {}
    for key in KEYS:
        matches = re.findall(r'^[ \t]*(?:export[ \t]+)?'+key+r'[ \t]*=(.*)$', text, flags=re.M | re.I)
        if len(matches) > 1:
            raise ValueError('Duplicate promotion configuration key: '+key)
        result[key] = matches[0].strip() if matches else None
    return result


def render_env(text: str, changes: dict[str, str | None]) -> str:
    if set(changes) != set(KEYS):
        raise ValueError('Promotion must bind exactly the generation keys')
    env_values(text)  # fail before overwriting ambiguous configuration
    if any(v is not None and ('\n' in v or '\r' in v) for v in changes.values()):
        raise ValueError('Multiline configuration is not permitted')
    newline = '\r\n' if '\r\n' in text else '\n'
    seen = set()
    result = []
    for line in text.splitlines(keepends=True):
        match = re.match(r'^[ \t]*(?:export[ \t]+)?([A-Z_]+)[ \t]*=', line, flags=re.I)
        key = match.group(1).upper() if match else None
        if key not in changes:
            result.append(line)
        else:
            seen.add(key)
            if changes[key] is not None:
                result.append(f'{key}={changes[key]}{newline}')
    for key in KEYS:
        if key not in seen and changes[key] is not None:
            if result and not result[-1].endswith(('\r', '\n')):
                result[-1] += newline
            result.append(f'{key}={changes[key]}{newline}')
    return ''.join(result)


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def verify_report(path: Path, *, artifact, lineage=None) -> dict:
    report = json.loads(path.read_bytes())
    meta = report['metadata']
    if (not report['summary']['release_gate_eligible'] or not meta['reviewed_manifest']
            or meta['code_artifact_identity_sha256'] != artifact.artifact_identity_sha256
            or meta['code_snapshot_id'] != artifact.snapshot_id):
        raise ValueError('Evaluation is failed, unreviewed, or for another code generation')
    if lineage and (meta['lineage_artifact_identity_sha256'] != lineage.artifact_identity_sha256
                    or meta['fdd_generation'] != lineage.fdd_generation):
        raise ValueError('Combined evaluation lineage/FDD identity mismatch')
    expected_mode = 'combined' if lineage else 'code'
    if not report['cases'] or any(c['mode'] != expected_mode or c['failures'] for c in report['cases']):
        raise ValueError('Evaluation contains failed cases or the wrong mode')
    for name, expected in meta['eval_file_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Reviewed evaluation manifest changed')
    return report


def verify_stores(settings, artifact, collection: str) -> dict:
    from qdrant_client import QdrantClient
    if not Path(settings.code_qdrant_local_path).is_dir() or not Path(settings.qdrant_local_path).is_dir():
        raise ValueError('Existing Qdrant store directories are required')
    code = QdrantClient(path=str(settings.code_qdrant_local_path))
    try:
        checked = verify_code_collection(code, collection_name=collection, artifact=artifact)
    finally:
        code.close()
    fdd = QdrantClient(path=str(settings.qdrant_local_path))
    try:
        if not fdd.collection_exists(settings.qdrant_collection_name):
            raise ValueError('Selected FDD collection is missing')
        info = fdd.get_collection(settings.qdrant_collection_name)
        if not info.points_count or info.config.params.vectors.size != artifact.vector_dimension:
            raise ValueError('FDD collection is empty or embedding dimensions differ')
        return {'code_verified_points': checked.verified_points, 'fdd_points': info.points_count}
    finally:
        fdd.close()


def prepare(*, root: Path, settings, code_artifact: Path, analysis: Path, lineage_path: Path,
            dependency_ledger: Path, code_report: Path, combined_report: Path,
            collection: str, requested_by: str) -> dict:
    root = root.resolve()
    if not requested_by.strip() or not re.fullmatch(r'code_custom_[A-Za-z0-9_]+', collection):
        raise ValueError('Requester and explicit code collection are required')
    artifact = load_code_index_artifact(code_artifact)
    if artifact.status != 'embedded' or artifact.dependency_review_status != 'reviewed':
        raise ValueError('Reviewed embedded artifact is required')
    ledger = DependencyReviewLedger.model_validate_json(dependency_ledger.read_bytes())
    if _ledger_identity(ledger) != ledger.ledger_identity_sha256:
        raise ValueError('Dependency ledger integrity failed')
    rebuilt = build_code_index_artifact(analysis, embedding_model=artifact.embedding_model, dependency_review_ledger=ledger)
    if rebuilt.artifact_identity_sha256 != artifact.artifact_identity_sha256:
        raise ValueError('Code artifact does not match its reviewed parse generation')
    for original, embedded in zip(rebuilt.records, artifact.records, strict=True):
        excluded = {'embedding_status', 'vector'}
        if original.model_dump(exclude=excluded) != embedded.model_dump(exclude=excluded):
            raise ValueError('Embedded code text/metadata differs from reviewed parse')
    if artifact.embedding_model != settings.openai_embedding_model:
        raise ValueError('Embedding model differs from candidate')
    lineage = load_reviewed_lineage(lineage_path)
    if lineage.fdd_generation != settings.fdd_generation or settings.fdd_generation != settings.qdrant_collection_name:
        raise ValueError('Selected FDD generation and lineage are inconsistent')
    documents = load_retrieval_ready_documents(settings.fdd_retrieval_artifact_dir)
    validate_lineage_artifact(lineage, code_artifact=artifact, analysis_directory=analysis,
                             fdd_document_ids={d.document_id for d in documents})
    cr = verify_report(code_report, artifact=artifact)
    combined = verify_report(combined_report, artifact=artifact, lineage=lineage)
    eval_dir = local(root, combined['metadata']['fdd_directory'])
    active_dir = settings.fdd_retrieval_artifact_dir
    def fdd_hashes(directory):
        return {p.name: sha(p) for p in directory.glob('*.retrieval_ready.json')}
    if not fdd_hashes(active_dir) or fdd_hashes(active_dir) != fdd_hashes(eval_dir):
        raise ValueError('Serving FDD units differ from the evaluated FDD units')
    stores = verify_stores(settings, artifact, collection)
    env = root/'.env'
    text = env.read_bytes().decode('utf-8')
    target = dict(zip(KEYS, ('true', str(local(root, code_artifact)), str(local(root, analysis)),
                             collection, str(local(root, lineage_path)))))
    # A prior code/FDD pair may be incompatible. Restore its selection DISABLED,
    # not an unverified claim that the previous combined runtime is healthy.
    rollback = {**env_values(text), 'CODE_MODES_ENABLED': 'false'}
    paths = [code_artifact, lineage_path, dependency_ledger, code_report, combined_report]
    paths += list(analysis.rglob('*.json')) + list(active_dir.glob('*.retrieval_ready.json'))
    paths += [Path(p) for r in (cr, combined) for p in r['metadata']['eval_file_sha256']]
    target_settings = Settings(**target)
    request = dict(schema_version=SCHEMA, created_at_utc=datetime.now(UTC).isoformat(),
                   requested_by=requested_by, status='pending_approval',
                   target_configuration=target, rollback_configuration=rollback,
                   before_env_sha256=sha(env), target_env_sha256=hashlib.sha256(render_env(text, target).encode()).hexdigest(),
                   before_settings_sha256=settings_identity(settings), target_settings_sha256=settings_identity(target_settings),
                   runtime_files=runtime_files(root), evidence_files=file_set(root, paths),
                   source_trees={str(local(root, analysis).relative_to(root)): '*.json',
                                 str(local(root, active_dir).relative_to(root)): '*.retrieval_ready.json'},
                   snapshot_id=artifact.snapshot_id, code_artifact_identity_sha256=artifact.artifact_identity_sha256,
                   lineage_identity_sha256=lineage.artifact_identity_sha256,
                   fdd_generation=settings.fdd_generation, store_checks=stores,
                   authorized_paid_requests=0, disclosure_authorized=False,
                   restart_required=True, activation_complete=False)
    request['request_identity_sha256'] = digest(request)
    return request


def verify_request(request: dict, root: Path) -> None:
    if request['schema_version'] != SCHEMA or digest({k:v for k,v in request.items() if k != 'request_identity_sha256'}) != request['request_identity_sha256']:
        raise ValueError('Promotion request integrity failed')
    if runtime_files(root) != request['runtime_files']:
        raise ValueError('Runtime changed; prepare a new request and approval')
    if file_set(root, [root/p for p in request['evidence_files']]) != request['evidence_files']:
        raise ValueError('Bound generation or evaluation evidence changed')
    for directory, pattern in request['source_trees'].items():
        observed = file_set(root, local(root, directory).rglob(pattern))
        recorded = {p:h for p,h in request['evidence_files'].items()
                    if Path(p).is_relative_to(Path(directory))}
        if observed != recorded:
            raise ValueError('Generation directory membership changed')


def switch(*, root: Path, request: dict, approval: ActivationApproval, action: str,
           settings, apply: bool = False) -> dict:
    root = root.resolve()
    if action not in {'activate', 'rollback'}:
        raise ValueError('Unknown promotion action')
    # Rollback must remain available if new runtime/evidence later becomes bad.
    if action == 'activate':
        verify_request(request, root)
    elif digest({k:v for k,v in request.items() if k != 'request_identity_sha256'}) != request['request_identity_sha256']:
        raise ValueError('Rollback request integrity failed')
    if (approval.request_identity_sha256 != request['request_identity_sha256']
            or approval.decision != 'approved' or action not in approval.allowed_actions
            or approval_identity(approval.model_dump(mode='json')) != approval.approval_identity_sha256):
        raise PermissionError('Exact generation-promotion approval is required')
    env = root/'.env'
    expected = request['before_env_sha256'] if action == 'activate' else request['target_env_sha256']
    if sha(env) != expected:
        raise ValueError('.env differs from the expected state; stop and reconcile')
    if action == 'activate':
        if settings_identity(settings) != request['before_settings_sha256']:
            raise ValueError('Effective settings changed, including environment overrides')
        for key, value in request['target_configuration'].items():
            if key in os.environ and os.environ[key] != value:
                raise ValueError('A process environment override conflicts with promotion: '+key)
        artifact = load_code_index_artifact(Path(request['target_configuration']['CODE_INDEX_ARTIFACT_PATH']))
        verify_stores(settings, artifact, request['target_configuration']['CODE_QDRANT_COLLECTION_NAME'])
    changes = request['target_configuration'] if action == 'activate' else request['rollback_configuration']
    if action == 'rollback':
        for key, value in changes.items():
            if key in os.environ and (value is None or os.environ[key] != value):
                raise ValueError('Remove process environment overrides before rollback: '+key)
    text = render_env(env.read_bytes().decode('utf-8'), changes)
    result = dict(action=action, request_identity_sha256=request['request_identity_sha256'],
                  applied=apply, restart_required=True, activation_complete=False,
                  external_api_calls=0, rollback_disables_code_modes=True)
    if apply:
        lock = root/'.code-generation-promotion.lock'
        handle = lock.open('x')
        try:
            with handle:
                if sha(env) != expected:
                    raise ValueError('.env changed during preflight')
                receipt = root/'data/exports/activation'/('generation-'+datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ'))
                write_new(receipt.with_suffix('.intent.json'), {**result, 'applied': False})
                _atomic_write(env, text)
                write_new(receipt.with_suffix('.result.json'), result)
        finally:
            lock.unlink()
    return result
