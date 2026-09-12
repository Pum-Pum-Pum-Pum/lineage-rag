import json

import pytest

from app.fdd_code_lineage.models import (
    FddCodeLineageArtifact, build_lineage_artifact, create_mapping, validate_lineage_artifact,
)
from app.fdd_code_lineage.reviewed_bundle import (
    build_bundle, digest, identity, load_evaluation_lineage, load_reviewed_lineage, validate_bundle, write_bundle,
)
from test_fdd_code_lineage import _code_artifact, _mapping, _analysis_directory, FDD_ID


def source_files(directory, label, *, fdd_generation='test_v1', newline='\n', fdd_id=None):
    directory.mkdir()
    code = _code_artifact()
    mapping = _mapping()
    candidate_mapping = create_mapping(fdd_document_id=fdd_id or FDD_ID+'_'+label,
        fdd_release_label='R22', code_snapshot_id=code.snapshot_id, targets=mapping.targets,
        rationale='Proposed supporting scope '+label)
    candidate = build_lineage_artifact(fdd_generation=fdd_generation, code_artifact=code,
        mappings=[candidate_mapping])
    rationale = 'Human accepts limited supporting scope '+label
    packet = newline.join(['# Review', f'- Candidate artifact: `{candidate.artifact_identity_sha256}`',
        '', '## 1. '+candidate_mapping.fdd_document_id, '', f'- Mapping ID: `{candidate_mapping.mapping_id}`',
        'SME verdict: reviewed', 'SME corrected targets/symbols:', 'SME rationale: '+rationale, ''])
    approved = create_mapping(fdd_document_id=candidate_mapping.fdd_document_id,
        fdd_release_label='R22', code_snapshot_id=code.snapshot_id, targets=mapping.targets,
        rationale=rationale, mapping_status='reviewed', reviewer='Pum')
    reviewed = build_lineage_artifact(fdd_generation=fdd_generation, code_artifact=code,
        mappings=[approved], source_candidate_artifact_identity_sha256=candidate.artifact_identity_sha256,
        review_packet_sha256=digest(packet), reviewer='Pum')
    paths = (directory/'reviewed.json', directory/'candidate.json', directory/'review.md')
    for path, content in zip(paths, (reviewed.model_dump_json(), candidate.model_dump_json(), packet)):
        path.write_bytes(content.encode('utf-8'))
    return paths


@pytest.fixture
def sources(tmp_path):
    return [source_files(tmp_path/'a', 'a', newline='\r\n'), source_files(tmp_path/'b', 'b')]


def rehash(payload):
    payload['artifact_identity_sha256'] = identity({k:v for k,v in payload.items() if k != 'artifact_identity_sha256'})
    return payload


def test_union_preserves_approvals_and_raw_bytes(sources, tmp_path):
    before = [[p.read_bytes() for p in paths] for paths in sources]
    payload = build_bundle(sources)
    bundle = validate_bundle(payload)
    assert bundle.status == 'reviewed' and len(bundle.mappings) == 2
    assert {m.reviewer for m in bundle.mappings} == {'Pum'}
    assert {m.mapping_id for m in bundle.mappings} == {m.mapping_id for s in bundle.sources for m in s.mappings}
    assert any('\r\n' in s['review_markdown'] for s in payload['sources'])
    assert [[p.read_bytes() for p in paths] for paths in sources] == before
    assert payload == build_bundle(list(reversed(sources)))
    path = tmp_path/'bundle.json'
    write_bundle(payload, path)
    assert load_evaluation_lineage(path) == bundle
    with pytest.raises(FileExistsError):
        write_bundle(payload, path)
    assert load_reviewed_lineage(path) == bundle
    # The raw v1 model still rejects bundles; serving dispatches explicitly.
    with pytest.raises(ValueError):
        FddCodeLineageArtifact.model_validate_json(path.read_text())
    summary = validate_lineage_artifact(bundle,
        fdd_document_ids={m.fdd_document_id for m in bundle.mappings},
        code_artifact=_code_artifact(), analysis_directory=_analysis_directory(tmp_path))
    assert summary['mappings'] == 2 and summary['targets'] == 2


@pytest.mark.parametrize('field', ['reviewed_json', 'candidate_json', 'review_markdown'])
def test_changed_retained_bytes_fail_even_with_rehashed_bundle(sources, field):
    payload = build_bundle(sources)
    payload['sources'][0][field] += ' '
    with pytest.raises(ValueError, match='byte hash'):
        validate_bundle(rehash(payload))


def test_changed_line_endings_do_not_reuse_approval(sources):
    packet = sources[0][2]
    packet.write_bytes(packet.read_bytes().replace(b'\r\n', b'\n'))
    with pytest.raises(ValueError, match='packet binding'):
        build_bundle(sources)


def test_cross_generation_union_rejected(tmp_path):
    paths = [source_files(tmp_path/'a', 'a'), source_files(tmp_path/'b', 'b', fdd_generation='different')]
    with pytest.raises(ValueError, match='different FDD/code'):
        build_bundle(paths)


def test_duplicate_parent_rejected(sources):
    with pytest.raises(ValueError, match='distinct reviewed'):
        build_bundle([sources[0], sources[0]])


def test_top_level_tampering_rejected(sources):
    payload = build_bundle(sources)
    payload['artifact_identity_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Bundle identity'):
        validate_bundle(payload)


def test_unapproved_candidate_cannot_be_source(sources):
    paths = list(sources)
    paths[0] = (paths[0][1], paths[0][1], paths[0][2])
    with pytest.raises(ValueError, match='reviewed and candidate'):
        build_bundle(paths)


def test_serving_loader_rejects_candidate_and_tampered_legacy_artifact(sources):
    reviewed, candidate, _ = sources[0]
    with pytest.raises(ValueError, match='Serving requires reviewed'):
        load_reviewed_lineage(candidate)
    payload = json.loads(reviewed.read_bytes())
    payload['artifact_identity_sha256'] = '0'*64
    reviewed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='identity mismatch'):
        load_reviewed_lineage(reviewed)


def test_rehashed_scope_change_cannot_masquerade_as_original_review(sources):
    payload = build_bundle(sources)
    source = payload['sources'][0]
    reviewed = json.loads(source['reviewed_json'])
    mapping = reviewed['mappings'][0]
    mapping['targets'][0]['rationale'] = 'Changed scope after original review'
    mapping['mapping_id'] = identity({k:v for k,v in mapping.items() if k != 'mapping_id'})
    reviewed['artifact_identity_sha256'] = identity({k:v for k,v in reviewed.items() if k not in {'schema_version','artifact_identity_sha256'}})
    source['reviewed_json'] = json.dumps(reviewed)
    source['byte_sha256']['reviewed_json'] = digest(source['reviewed_json'])
    with pytest.raises(ValueError, match='differ from their candidate'):
        validate_bundle(rehash(payload))


def test_blank_rationale_cannot_consume_next_line(sources):
    # Update the packet hash and artifact identity too, proving semantic checks run.
    payload = build_bundle(sources)
    source = payload['sources'][0]
    packet = source['review_markdown']
    start = packet.index('SME rationale:')
    packet = packet[:start] + 'SME rationale:\nSME verdict: reviewed\n'
    source['review_markdown'] = packet
    source['byte_sha256']['review_markdown'] = digest(packet)
    reviewed = json.loads(source['reviewed_json'])
    reviewed['review_packet_sha256'] = digest(packet)
    reviewed['artifact_identity_sha256'] = identity({k:v for k,v in reviewed.items() if k not in {'schema_version','artifact_identity_sha256'}})
    source['reviewed_json'] = json.dumps(reviewed)
    source['byte_sha256']['reviewed_json'] = digest(source['reviewed_json'])
    with pytest.raises(ValueError, match='duplicate review field|unresolved'):
        validate_bundle(rehash(payload))


def test_legacy_evaluation_loader_unchanged(sources):
    assert load_evaluation_lineage(sources[0][0]).status == 'reviewed'
