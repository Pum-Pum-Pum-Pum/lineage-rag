from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.embeddings.embedding_contract import compute_content_hash, compute_embedding_cache_key
from app.fdd_code_lineage.enhancement_comments import extract_regions, identity_key
from app.fdd_code_lineage import semantic_proposals as proposals
from app.code_ingestion.code_analysis_models import CodeStaticAnalysisArtifact
from app.code_ingestion.plsql_models import CodeParseStageManifest
from test_fdd_code_lineage import _code_artifact, _analysis_directory


MARKER = 'FCIS_14.7.0.0.0$ASNB_R24 (REQ07) - NEO Day2 Part2 Enhancement'


def test_comment_regions_are_exact_nested_and_ignore_sql_strings():
    source = (f"select '--{MARKER} :: Start' from dual;\r\n"
              f"--{MARKER} :: Developer :: Date :: Start\r\n"
              f"--{MARKER} SCR :: Start\r\nNULL;\r\n"
              f"--{MARKER} SCR :: End\r\n--{MARKER} :: End\r\n")
    regions, mentions, diagnostics = extract_regions(source)
    assert len(regions) == 2
    assert regions[0].start_line == 2 and regions[0].end_line == 6
    assert regions[1].start_line == 3 and regions[1].end_line == 5
    assert source[regions[0].start_offset:regions[0].end_offset].startswith('--FCIS')
    assert not mentions and not diagnostics
    assert regions[0].identity != regions[1].identity
    assert 'Developer' not in str(regions)


def test_unpaired_and_crossed_markers_do_not_create_regions():
    regions, _, diagnostics = extract_regions(
        f'--{MARKER} :: Start\n--{MARKER} SCR :: Start\n--{MARKER} :: End\n')
    assert not regions and diagnostics
    regions, mentions, _ = extract_regions('--FCIS_14.7.0.0.0$ASNB_R66 - Neo Day2\n')
    assert not regions and len(mentions) == 1
    assert mentions[0]['identity'] != identity_key('FCIS_14.7.0.0.0$ASNB_R24', 'REQ07', 'NEO Day2 Part2 Enhancement')


@pytest.mark.parametrize('vectors', [[[0, 0]], [[float('nan'), 1]], [[1, 2, 3]], [[float('inf'), 1]]])
def test_invalid_vectors_fail(vectors):
    with pytest.raises(ValueError):
        proposals.unit_vectors(vectors, 2)


def test_cosine_normalization():
    matrix = proposals.unit_vectors([[3, 4], [0, 5]], 2)
    assert matrix[0] @ matrix[1] == pytest.approx(.8)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1, 1])


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    def dump(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    code = _code_artifact()
    source = ' ' * 100 + f'--{MARKER} :: Start\nNULL;\n--{MARKER} :: End\n'
    sm = code.records[0].source_map.model_copy(update={'end_offset': len(source)})
    record = code.records[0].model_copy(update={
        'source_map': sm, 'citation_text': source[100:],
        'content_sha256': hashlib.sha256(code.records[0].embedding_text.encode()).hexdigest()})
    code = code.model_copy(update={'records': (record,)})
    artifact = tmp_path / 'code.json'
    dump(artifact, code.model_dump(mode='json'))
    monkeypatch.setattr(proposals, 'load_code_index_artifact', lambda _: code)
    snapshot = tmp_path / 'snapshot'
    dump(snapshot / 'snapshot_manifest.json', {'fixture': True})
    raw = source.encode()
    (snapshot / 'source').mkdir()
    (snapshot / 'source' / record.source_path).write_bytes(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(proposals, 'load_snapshot_manifest', lambda _: SimpleNamespace(
        snapshot_id=code.snapshot_id, snapshot_content_sha256=code.snapshot_content_sha256,
        files=[SimpleNamespace(path=record.source_path, sha256=source_hash, encoding='utf-8')]))
    analysis_dir = _analysis_directory(tmp_path)
    analysis_path = analysis_dir / 'aml.json'
    analysis = CodeStaticAnalysisArtifact.model_validate_json(analysis_path.read_text())
    analysis = analysis.model_copy(update={'source_sha256': source_hash,
        'symbols': (analysis.symbols[0].model_copy(update={'source_map': sm}),)})
    dump(analysis_path, analysis.model_dump(mode='json'))
    parsed = CodeParseStageManifest(status='complete', snapshot_id=code.snapshot_id,
        snapshot_content_sha256=code.snapshot_content_sha256, parser_generation=code.parse_generation,
        analysis_policy_sha256=code.analysis_policy_sha256, file_count=1,
        state_counts=dict(full_parse=1, segmented_parse=0, fallback_parse=0, failed=0),
        parse_artifacts=('p.json',), retrieval_artifacts=('r.json',), analysis_artifacts=('aml.json',),
        timeout_seconds=1, memory_limit_bytes=1000, max_segment_characters=500,
        max_retrieval_unit_characters=6000, retrieval_overlap_characters=400)
    dump(analysis_dir / 'parse_stage_manifest.json', parsed.model_dump(mode='json'))
    fdd_stage = tmp_path / 'fdd'
    dump(fdd_stage / 'stage_manifest.json', dict(status='verified', index_generation='test_v1',
        sources=[dict(document_name=name+'.docx') for name in ('fdd_a', 'fdd_b', 'fdd_c')],
        embedding_model=code.embedding_model, embedding_record_artifact_version='v1'))
    for name, vector in [('fdd_a', [1, 0]), ('fdd_b', [0, 1]), ('fdd_c', [-1, 0])]:
        text = f'Evidence for {name} <script>not executable</script>'
        dump(fdd_stage / 'processed' / (name+'.retrieval_ready.json'), dict(document_id=name,
            units=[dict(unit_id='u1', text=text, retrieval_text=text)]))
        content_hash = compute_content_hash(text)
        dump(fdd_stage / 'cache' / 'embeddings' / (name+'.embeddings.json'), dict(
            document_name=name+'.docx', records=[dict(unit_id='u1', document_id=name, text=text,
            source_text=text, content_hash=content_hash, artifact_version='v1',
            cache_key=compute_embedding_cache_key(content_hash, code.embedding_model, 'v1'),
            embedding_model=code.embedding_model, embedding_status='embedded', vector=vector,
            source_kind='paragraph')]))
    registry = tmp_path / 'registry.json'
    dump(registry, dict(schema_version='enhancement_fdd_registry_v1', mappings=[dict(
        code_release='FCIS_14.7.0.0.0$ASNB_R24', requirement='REQ07',
        title='NEO Day2 Part2 Enhancement', fdd_document_id='fdd_b', basis='Test reviewer clarification')]))
    return dict(fdd_stage=fdd_stage, snapshot_directory=snapshot, analysis_directory=analysis_dir,
                code_artifact_path=artifact, registry_path=registry)


def test_local_pipeline_candidates_do_not_approve_or_force_matches(inputs, monkeypatch, tmp_path):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *a: pytest.fail('Network is forbidden'))
    result = proposals.build_proposals(**inputs)
    assert result['external_api_calls'] == result['automatic_approvals'] == 0
    a, b, c = result['documents']
    assert a['candidates'][0]['basis'] == 'similarity_only'
    assert a['candidates'][0]['cosine_similarity'] == 1
    assert b['candidates'][0]['basis'] == 'comment_region_and_similarity'
    assert c['status'] == 'no_strong_candidate'
    assert result == proposals.build_proposals(**inputs)
    assert result['input_sha256']
    html = proposals.render_report(result)
    assert '<script>not executable</script>' not in html
    assert '&lt;script&gt;not executable&lt;/script&gt;' in html
    output = tmp_path / 'report'
    proposals.write_report(result, output)
    with pytest.raises(FileExistsError):
        proposals.write_report(result, output)


@pytest.mark.parametrize('field,value', [('text', 'altered'), ('embedding_model', 'different'),
                                       ('cache_key', '0'*64), ('document_id', 'wrong')])
def test_stale_fdd_embeddings_fail_closed(inputs, field, value):
    path = inputs['fdd_stage'] / 'cache/embeddings/fdd_a.embeddings.json'
    batch = json.loads(path.read_text())
    batch['records'][0][field] = value
    path.write_text(json.dumps(batch))
    with pytest.raises(ValueError):
        proposals.build_proposals(**inputs)


def test_changed_source_fails_closed(inputs):
    (inputs['snapshot_directory'] / 'source/pkgaml_custom.sql').write_text('changed')
    with pytest.raises(ValueError, match='Snapshot source changed'):
        proposals.build_proposals(**inputs)


def test_directory_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        proposals.under(tmp_path, '../outside.json')
