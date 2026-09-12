"""Provenance-preserving union of independently reviewed lineage mappings.

Original UTF-8 artifact/packet bytes are embedded, not replaced by a synthetic
reviewer or a new claim of SME acceptance. Serving uses an explicit verified loader.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from app.fdd_code_lineage.models import FddCodeLineageArtifact, create_mapping

SCHEMA = 'fdd_code_reviewed_bundle_v1'
OPERATION = 'union_existing_reviewed_mappings'


def digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def identity(value: dict) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False))


def verify_artifact(text: str) -> FddCodeLineageArtifact:
    artifact = FddCodeLineageArtifact.model_validate_json(text)
    fields = artifact.model_dump(mode='json', exclude={'schema_version', 'artifact_identity_sha256'})
    if identity(fields) != artifact.artifact_identity_sha256:
        raise ValueError('Lineage artifact identity mismatch')
    for mapping in artifact.mappings:
        if identity(mapping.model_dump(mode='json', exclude={'mapping_id'})) != mapping.mapping_id:
            raise ValueError('Lineage mapping identity mismatch')
    return artifact


def verify_source(source: dict) -> FddCodeLineageArtifact:
    names = {'reviewed_json', 'candidate_json', 'review_markdown'}
    if set(source) != names | {'byte_sha256'} or set(source['byte_sha256']) != names:
        raise ValueError('Invalid review source fields')
    for name in names:
        if digest(source[name]) != source['byte_sha256'][name]:
            raise ValueError('Retained review source byte hash mismatch')
    reviewed = verify_artifact(source['reviewed_json'])
    candidate = verify_artifact(source['candidate_json'])
    packet = source['review_markdown']
    if reviewed.status != 'reviewed' or candidate.status != 'candidate':
        raise ValueError('Bundle sources require reviewed and candidate artifacts')
    generation = lambda a: (a.fdd_generation, a.code_snapshot_id, a.code_artifact_identity_sha256)
    if generation(reviewed) != generation(candidate):
        raise ValueError('Candidate/reviewed generation mismatch')
    if reviewed.source_candidate_artifact_identity_sha256 != candidate.artifact_identity_sha256:
        raise ValueError('Reviewed source candidate binding mismatch')
    if reviewed.review_packet_sha256 != digest(packet):
        raise ValueError('Reviewed packet binding mismatch')
    if f'Candidate artifact: `{candidate.artifact_identity_sha256}`' not in packet:
        raise ValueError('Packet candidate binding mismatch')
    sections = re.finditer(r'^## \d+\. [^\r\n]+\r?\n(?P<body>.*?)(?=^## \d+\.|\Z)', packet, re.M | re.S)
    decisions = {}
    for section in sections:
        body = section['body']
        def field(pattern):
            values = re.findall(pattern, body, re.M)
            if len(values) != 1:
                raise ValueError('Missing or duplicate review field')
            return values[0].strip()
        mapping_id = field(r'^- Mapping ID: `([0-9a-f]{64})`[ \t]*\r?$')
        verdict = field(r'^SME verdict:[ \t]*([^\r\n]*)\r?$')
        rationale = field(r'^SME rationale:[ \t]*([^\r\n]*)\r?$')
        corrections = field(r'^SME corrected targets/symbols:[ \t]*([^\r\n]*)\r?$')
        if verdict != 'reviewed' or not rationale or corrections:
            raise ValueError('Review is unresolved or contains unapplied target corrections')
        if mapping_id in decisions:
            raise ValueError('Duplicate mapping decision')
        decisions[mapping_id] = rationale
    if set(decisions) != {m.mapping_id for m in candidate.mappings}:
        raise ValueError('Review decisions do not match candidate mappings')
    # Reconstruct the exact original import, including human rationale/reviewer.
    expected = [create_mapping(fdd_document_id=m.fdd_document_id,
        fdd_release_label=m.fdd_release_label, code_snapshot_id=m.code_snapshot_id,
        targets=m.targets, rationale=decisions[m.mapping_id], mapping_status='reviewed',
        reviewer=reviewed.reviewer) for m in candidate.mappings]
    if sorted(expected, key=lambda m: m.mapping_id) != list(reviewed.mappings):
        raise ValueError('Reviewed mappings differ from their candidate/SME decisions')
    return reviewed


@dataclass(frozen=True)
class ReviewedLineageBundle:
    """Read-only structural interface consumed by existing retrieval validation."""
    sources: tuple[FddCodeLineageArtifact, ...]
    artifact_identity_sha256: str
    schema_version: str = SCHEMA

    @property
    def status(self):
        return 'reviewed'

    @property
    def fdd_generation(self):
        return self.sources[0].fdd_generation

    @property
    def code_snapshot_id(self):
        return self.sources[0].code_snapshot_id

    @property
    def code_artifact_identity_sha256(self):
        return self.sources[0].code_artifact_identity_sha256

    @property
    def mappings(self):
        found = {}
        for source in self.sources:
            for mapping in source.mappings:
                if mapping.mapping_id in found and found[mapping.mapping_id] != mapping:
                    raise ValueError('Conflicting mapping identity')
                found[mapping.mapping_id] = mapping
        return tuple(found[key] for key in sorted(found))


def validate_bundle(payload: dict) -> ReviewedLineageBundle:
    if set(payload) != {'schema_version', 'operation', 'sources', 'artifact_identity_sha256'}:
        raise ValueError('Invalid bundle fields')
    if payload['schema_version'] != SCHEMA or payload['operation'] != OPERATION:
        raise ValueError('Unknown bundle contract')
    expected = identity({k: v for k, v in payload.items() if k != 'artifact_identity_sha256'})
    if expected != payload['artifact_identity_sha256']:
        raise ValueError('Bundle identity mismatch')
    sources = tuple(verify_source(source) for source in payload['sources'])
    if len(sources) < 2 or len({s.artifact_identity_sha256 for s in sources}) != len(sources):
        raise ValueError('At least two distinct reviewed sources are required')
    generations = {(s.fdd_generation, s.code_snapshot_id, s.code_artifact_identity_sha256) for s in sources}
    if len(generations) != 1:
        raise ValueError('Cannot consolidate different FDD/code generations')
    bundle = ReviewedLineageBundle(sources, expected)
    bundle.mappings  # Validate duplicate handling before use.
    return bundle


def build_bundle(source_paths: list[tuple[Path, Path, Path]]) -> dict:
    sources = []
    for reviewed, candidate, packet in source_paths:
        # Decode bytes explicitly: text-mode reads would normalize CRLF approval bytes.
        source = dict(reviewed_json=reviewed.read_bytes().decode('utf-8'),
                      candidate_json=candidate.read_bytes().decode('utf-8'),
                      review_markdown=packet.read_bytes().decode('utf-8'))
        source['byte_sha256'] = {name: digest(value) for name, value in source.items()}
        verify_source(source)
        sources.append(source)
    sources.sort(key=lambda s: s['byte_sha256']['reviewed_json'])
    payload = dict(schema_version=SCHEMA, operation=OPERATION, sources=sources)
    payload['artifact_identity_sha256'] = identity(payload)
    validate_bundle(payload)
    return payload


def write_bundle(payload: dict, path: Path) -> None:
    validate_bundle(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)


def load_evaluation_lineage(path: Path) -> FddCodeLineageArtifact | ReviewedLineageBundle:
    text = path.read_bytes().decode('utf-8')
    payload = json.loads(text)
    if payload.get('schema_version') == SCHEMA:
        return validate_bundle(payload)
    # Preserve the pre-existing v1 evaluation loader contract.
    return FddCodeLineageArtifact.model_validate_json(text)


def load_reviewed_lineage(path: Path) -> FddCodeLineageArtifact | ReviewedLineageBundle:
    """Serving boundary: validate integrity and reject unreviewed lineage.

    Bundles retain and verify both complete original approval chains. The legacy
    v1 format retains its existing candidate/packet identities, without pretending
    it contains the original packet bytes.
    """
    text = path.read_bytes().decode('utf-8')
    payload = json.loads(text)
    artifact = (validate_bundle(payload) if payload.get('schema_version') == SCHEMA
                else verify_artifact(text))
    if artifact.status != 'reviewed':
        raise ValueError('Serving requires reviewed lineage')
    return artifact
