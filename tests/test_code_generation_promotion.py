from pathlib import Path
from types import SimpleNamespace

import pytest

from app.activation import code_generation as promotion
from app.activation.code_modes import build_activation_approval


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path/'app').mkdir()
    (tmp_path/'app/runtime.py').write_text('original runtime')
    (tmp_path/'scripts').mkdir()
    (tmp_path/'pyproject.toml').write_text('project')
    (tmp_path/'uv.lock').write_text('lock')
    (tmp_path/'evidence.json').write_text('reviewed evidence')
    env = tmp_path/'.env'
    env.write_bytes(b'OPENAI_API_KEY=test-secret\r\n# preserve me\r\nCODE_MODES_ENABLED=true\r\n')
    target = dict(zip(promotion.KEYS, ('true', 'new-code.json', 'analysis', 'code_custom_new', 'lineage.json')))
    before = env.read_bytes().decode()
    settings = SimpleNamespace(model_dump=lambda **_: {'interface_mode': 'mcp'})
    request = dict(schema_version=promotion.SCHEMA, status='pending_approval',
                   target_configuration=target,
                   rollback_configuration={**promotion.env_values(before), 'CODE_MODES_ENABLED':'false'},
                   before_env_sha256=promotion.sha(env),
                   target_env_sha256=promotion.hashlib.sha256(promotion.render_env(before,target).encode()).hexdigest(),
                   before_settings_sha256=promotion.settings_identity(settings),
                   evidence_files=promotion.file_set(tmp_path,[tmp_path/'evidence.json']),
                   source_trees={}, runtime_files=promotion.runtime_files(tmp_path))
    request['request_identity_sha256'] = promotion.digest(request)
    approval = build_activation_approval(request=SimpleNamespace(**request), approved_by='Tester')
    monkeypatch.setattr(promotion, 'load_code_index_artifact', lambda _: object())
    monkeypatch.setattr(promotion, 'verify_stores', lambda *args: {})
    for key in promotion.KEYS:
        monkeypatch.delenv(key, raising=False)
    return tmp_path, request, approval, settings


def invoke(setup, **kwargs):
    root, request, approval, settings = setup
    return promotion.switch(root=root, request=request, approval=approval, settings=settings,
                            action=kwargs.pop('action', 'activate'), **kwargs)


def test_dry_run_has_zero_writes(setup):
    root, *_ = setup
    before = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert invoke(setup)['applied'] is False
    assert {p:p.read_bytes() for p in root.rglob('*') if p.is_file()} == before


def test_apply_and_safe_rollback_preserve_unrelated_env_and_missing_keys(setup):
    root, request, *_ = setup
    assert invoke(setup, apply=True)['activation_complete'] is False
    text = (root/'.env').read_bytes().decode()
    assert 'OPENAI_API_KEY=test-secret\r\n# preserve me\r\n' in text
    assert promotion.env_values(text) == request['target_configuration']
    assert promotion.sha(root/'.env') == request['target_env_sha256']
    assert invoke(setup, action='rollback', apply=True)['applied'] is True
    assert (root/'.env').read_bytes() == b'OPENAI_API_KEY=test-secret\r\n# preserve me\r\nCODE_MODES_ENABLED=false\r\n'
    assert not (root/'.code-generation-promotion.lock').exists()
    assert len(list((root/'data/exports/activation').glob('*.result.json'))) == 2
    for p in (root/'data/exports/activation').glob('*.json'):
        assert 'test-secret' not in p.read_text()


@pytest.mark.parametrize('which', ['request', 'approval', 'runtime', 'evidence', 'env', 'new-runtime-file'])
def test_stale_or_altered_inputs_block_apply(setup, which):
    root, request, approval, settings = setup
    if which == 'request':
        request['target_configuration']['CODE_QDRANT_COLLECTION_NAME'] = 'other'
    elif which == 'approval':
        setup = root, request, approval.model_copy(update={'approved_by':'changed'}), settings
    else:
        path = {'runtime':'app/runtime.py', 'evidence':'evidence.json', 'env':'.env',
                'new-runtime-file':'app/new.py'}[which]
        (root/path).write_text('changed')
    before = (root/'.env').read_bytes()
    with pytest.raises((ValueError, PermissionError)):
        invoke(setup, apply=True)
    assert (root/'.env').read_bytes() == before


def test_process_environment_override_blocks_promotion(setup, monkeypatch):
    monkeypatch.setenv('CODE_QDRANT_COLLECTION_NAME','code_custom_stale')
    with pytest.raises(ValueError, match='override'):
        invoke(setup)


def test_qdrant_failure_does_not_change_env(setup, monkeypatch):
    def fail(*args):
        raise RuntimeError('locked')
    monkeypatch.setattr(promotion, 'verify_stores', fail)
    root, *_ = setup
    before = (root/'.env').read_bytes()
    with pytest.raises(RuntimeError, match='locked'):
        invoke(setup, apply=True)
    assert (root/'.env').read_bytes() == before


def test_rollback_cannot_be_overridden_by_process_flag(setup, monkeypatch):
    invoke(setup, apply=True)
    monkeypatch.setenv('CODE_MODES_ENABLED', 'true')
    with pytest.raises(ValueError, match='overrides before rollback'):
        invoke(setup, action='rollback', apply=True)


def test_write_failure_leaves_intent_and_releases_owned_lock(setup, monkeypatch):
    root, *_ = setup
    before = (root/'.env').read_bytes()
    def fail(*args):
        raise OSError('injected write failure')
    monkeypatch.setattr(promotion, '_atomic_write', fail)
    with pytest.raises(OSError):
        invoke(setup, apply=True)
    assert (root/'.env').read_bytes() == before
    assert not (root/'.code-generation-promotion.lock').exists()
    assert len(list((root/'data/exports/activation').glob('*.intent.json'))) == 1


def test_rollback_remains_available_after_new_runtime_breaks(setup):
    invoke(setup, apply=True)
    root, *_ = setup
    (root/'app/runtime.py').write_text('broken runtime')
    assert invoke(setup, action='rollback', apply=True)['applied']


def test_concurrent_promotion_lock_fails_without_removing_other_lock(setup):
    root, *_ = setup
    lock = root/'.code-generation-promotion.lock'
    lock.write_text('other owner')
    with pytest.raises(FileExistsError):
        invoke(setup, apply=True)
    assert lock.read_text() == 'other owner'


def test_duplicate_keys_and_injected_lines_are_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        promotion.env_values('CODE_MODES_ENABLED=true\ncode_modes_enabled=false\n')
    with pytest.raises(ValueError, match='Multiline'):
        promotion.render_env('', {k:'x\nSECRET=x' for k in promotion.KEYS})


def test_write_new_refuses_overwrite(tmp_path):
    path = tmp_path/'request.json'
    promotion.write_new(path, {'approval':False})
    with pytest.raises(FileExistsError):
        promotion.write_new(path, {'approval':True})


def test_settings_fingerprint_excludes_secrets():
    from app.core.config import Settings
    a = Settings(_env_file=None, OPENAI_API_KEY='one')
    b = Settings(_env_file=None, OPENAI_API_KEY='two')
    assert promotion.settings_identity(a) == promotion.settings_identity(b)


@pytest.mark.parametrize('failure', ['gate', 'generation', 'case', 'manifest', 'lineage'])
def test_preparation_rejects_invalid_evaluation_evidence(tmp_path, failure):
    import json
    manifest = tmp_path/'reviewed.jsonl'
    manifest.write_text('reviewed cases')
    artifact = SimpleNamespace(artifact_identity_sha256='a'*64, snapshot_id='snapshot-r2')
    lineage = SimpleNamespace(artifact_identity_sha256='b'*64, fdd_generation='fdd-v9')
    report = {'summary': {'release_gate_eligible': True},
              'metadata': {'reviewed_manifest':True, 'code_artifact_identity_sha256':'a'*64,
                           'code_snapshot_id':'snapshot-r2', 'lineage_artifact_identity_sha256':'b'*64,
                           'fdd_generation':'fdd-v9', 'eval_file_sha256':{str(manifest):promotion.sha(manifest)}},
              'cases':[{'case_id':'regression-001', 'mode':'combined', 'failures':[]}]}
    if failure == 'gate': report['summary']['release_gate_eligible'] = False
    if failure == 'generation': report['metadata']['code_snapshot_id'] = 'other'
    if failure == 'case': report['cases'][0]['failures'] = ['missing evidence']
    if failure == 'manifest': manifest.write_text('altered cases')
    if failure == 'lineage': report['metadata']['lineage_artifact_identity_sha256'] = 'c'*64
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='regression-001: missing evidence' if failure == 'case' else None):
        promotion.verify_report(path, artifact=artifact, lineage=lineage)
