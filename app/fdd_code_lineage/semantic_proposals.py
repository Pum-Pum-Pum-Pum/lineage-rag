"""Local evidence-assisted lineage discovery. All outputs are unapproved proposals."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Callable

import numpy as np

from app.code_indexing.contract import load_code_index_artifact
from app.code_ingestion.code_analysis_models import CodeStaticAnalysisArtifact
from app.code_ingestion.plsql_models import CodeParseStageManifest
from app.code_ingestion.snapshot_builder import load_snapshot_manifest
from app.embeddings.embedding_contract import compute_content_hash, compute_embedding_cache_key
from app.fdd_code_lineage.enhancement_comments import extract_regions, identity_key


SCHEMA = 'fdd_code_semantic_proposals_v1'


def read_bound(path: Path, bindings: dict[str, str], label: str) -> bytes:
    data = path.read_bytes()
    bindings[label] = hashlib.sha256(data).hexdigest()
    return data


def under(root: Path, relative: str) -> Path:
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Artifact reference escapes its input directory')
    return resolved


def unit_vectors(vectors, dimension: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension or not np.isfinite(matrix).all():
        raise ValueError('Embedding vectors have incompatible dimensions or nonfinite values')
    norms = np.linalg.norm(matrix, axis=1)
    if (norms == 0).any():
        raise ValueError('Zero embedding vector cannot be compared')
    return matrix / norms[:, None]


def load_registry(
    path: Path, document_ids: set[str], bindings: dict
) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]], set[str]]:
    registry = json.loads(read_bound(path, bindings, 'enhancement_registry'))
    if registry.get('schema_version') != 'enhancement_fdd_registry_v1':
        raise ValueError('Unknown enhancement registry schema')
    lookup: dict[str, set[str]] = {}
    scoped_lookup: dict[tuple[str, str], set[str]] = {}
    registered_identities: set[str] = set()
    for row in registry['mappings']:
        if not row.get('basis', '').strip():
            raise ValueError('Enhancement correspondence needs an attributable basis')
        doc = row['fdd_document_id']
        if doc not in document_ids:
            raise ValueError(f'Registry FDD is absent from selected generation: {doc}')
        key = identity_key(row['code_release'], row.get('requirement'), row['title'])
        registered_identities.add(key)
        source_paths = row.get('source_paths')
        if source_paths is None:
            lookup.setdefault(key, set()).add(doc)
            continue
        if not isinstance(source_paths, list) or not source_paths or any(
            not isinstance(item, str) or not item.strip() for item in source_paths
        ):
            raise ValueError('Scoped registry source_paths must be a nonempty string list')
        for source_path in source_paths:
            scoped_lookup.setdefault((key, source_path.replace('\\', '/')), set()).add(doc)
    return lookup, scoped_lookup, registered_identities


def build_proposals(*, fdd_stage: Path, snapshot_directory: Path, analysis_directory: Path,
                    code_artifact_path: Path, registry_path: Path, top_k: int = 5,
                    minimum_similarity: float = 0.45, ambiguity_margin: float = 0.03,
                    progress: Callable[[str], None] = lambda _: None,
                    allow_provisional: bool = False) -> dict:
    if not 1 <= top_k <= 20 or not 0 <= minimum_similarity <= 1 or not 0 <= ambiguity_margin <= 1:
        raise ValueError('Invalid candidate count or similarity thresholds')
    bindings: dict[str, str] = {}
    stage = json.loads(read_bound(fdd_stage / 'stage_manifest.json', bindings, 'fdd_stage_manifest'))
    if stage['status'] != 'verified':
        raise ValueError('FDD generation must be verified')
    source_names = [item['document_name'] for item in stage['sources']]
    if len(set(source_names)) != len(source_names):
        raise ValueError('Duplicate FDD source declarations')
    document_ids = {name.rsplit('.', 1)[0] for name in source_names}
    registry, scoped_registry, registered_identities = load_registry(
        registry_path, document_ids, bindings
    )
    read_bound(code_artifact_path, bindings, 'code_embedding_artifact')
    code = load_code_index_artifact(code_artifact_path)
    if code.status != 'embedded' or (code.dependency_review_status != 'reviewed' and not allow_provisional):
        raise ValueError('Code requires a reviewed embedded artifact')
    if code.embedding_model != stage['embedding_model']:
        raise ValueError('FDD and code embedding models differ; vectors cannot be compared')
    read_bound(snapshot_directory / 'snapshot_manifest.json', bindings, 'snapshot_manifest')
    snapshot = load_snapshot_manifest(snapshot_directory)
    if (snapshot.snapshot_id, snapshot.snapshot_content_sha256) != (code.snapshot_id, code.snapshot_content_sha256):
        raise ValueError('Code embedding artifact and immutable snapshot differ')
    parsed = CodeParseStageManifest.model_validate_json(read_bound(
        analysis_directory / 'parse_stage_manifest.json', bindings, 'parse_stage_manifest'))
    if (parsed.status == 'failed' or parsed.snapshot_id != code.snapshot_id or
        parsed.snapshot_content_sha256 != code.snapshot_content_sha256 or
        parsed.parser_generation != code.parse_generation or
        parsed.analysis_policy_sha256 != code.analysis_policy_sha256):
        raise ValueError('Code embedding and parse generations differ')
    entries = {entry.path: entry for entry in snapshot.files}
    sources: dict[str, str] = {}
    regions_by_path = {}
    marker_report = []
    symbols = []
    for relative in parsed.analysis_artifacts:
        analysis = CodeStaticAnalysisArtifact.model_validate_json(read_bound(
            under(analysis_directory, relative), bindings, f'analysis/{relative}'))
        entry = entries.get(analysis.source_path)
        if (entry is None or analysis.source_sha256 != entry.sha256 or
            analysis.snapshot_id != code.snapshot_id or analysis.analysis_policy_sha256 != code.analysis_policy_sha256):
            raise ValueError('Static analysis is not bound to selected source')
        if analysis.source_path in sources:
            raise ValueError('Duplicate source analysis')
        raw = read_bound(under(snapshot_directory / 'source', entry.path), bindings, f'source/{entry.path}')
        if hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise ValueError('Snapshot source changed during analysis')
        text = raw.decode(entry.encoding)
        sources[entry.path] = text
        progress(f'Extracting comments: {entry.path}')
        regions, mentions, diagnostics = extract_regions(text)
        regions_by_path[entry.path] = regions
        marker_report.append(dict(source_path=entry.path, regions=[r.as_dict() for r in regions],
                                  file_mentions=mentions, diagnostics=diagnostics))
        symbols.extend(analysis.symbols)
    if set(sources) != set(entries):
        raise ValueError('Analysis does not cover the complete snapshot')
    # Prefer implementation occurrences. A declaration cannot establish executable behavior.
    implementations = [s for s in symbols if s.occurrence_role == 'implementation']
    record_groups: dict[int, list[int]] = {}
    for index, record in enumerate(code.records):
        text = sources.get(record.source_path)
        sm = record.source_map
        if (record.snapshot_id != code.snapshot_id or record.embedding_model != code.embedding_model or
            text is None or sm.source_path != record.source_path or
            text[sm.start_offset:sm.end_offset] != record.citation_text or
            hashlib.sha256(record.embedding_text.encode('utf-8')).hexdigest() != record.content_sha256):
            raise ValueError('Code record failed exact source/content validation')
        matches = [(j, s) for j, s in enumerate(implementations)
                   if s.source_path == record.source_path and
                   s.source_map.start_offset < sm.end_offset and sm.start_offset < s.source_map.end_offset]
        containing = [(j, s) for j, s in matches if s.source_map.start_offset <= sm.start_offset
                      and s.source_map.end_offset >= sm.end_offset]
        if containing:
            smallest = min(s.source_map.end_offset - s.source_map.start_offset for _, s in containing)
            matches = [(j, s) for j, s in containing if s.source_map.end_offset - s.source_map.start_offset == smallest]
        if len(matches) == 1:
            record_groups.setdefault(matches[0][0], []).append(index)
    if not record_groups:
        raise ValueError('No code embedding units resolve unambiguously to implementation routines')
    code_vectors = unit_vectors([r.vector for r in code.records], code.vector_dimension)
    documents = []
    diagnostics = []
    vector_files = fdd_stage / 'cache' / 'embeddings'
    for number, name in enumerate(sorted(source_names), 1):
        stem = name.rsplit('.', 1)[0]
        progress(f'Comparing FDD {number}/{len(source_names)}: {stem}')
        ready = json.loads(read_bound(under(fdd_stage / 'processed', stem + '.retrieval_ready.json'),
                                     bindings, f'fdd_processed/{stem}'))
        if ready.get('document_id') != stem:
            raise ValueError('FDD processed document identity mismatch')
        units = {unit['unit_id']: unit for unit in ready['units']}
        if len(units) != len(ready['units']):
            raise ValueError('Duplicate FDD unit identities')
        embedding_file = under(vector_files, stem + '.embeddings.json')
        if not embedding_file.is_file():
            documents.append(dict(document_id=stem, status='missing_embeddings', candidates=[]))
            continue
        batch = json.loads(read_bound(embedding_file, bindings, f'fdd_embeddings/{stem}'))
        if batch['document_name'] != name:
            raise ValueError('FDD embedding batch identity mismatch')
        usable = []
        seen = set()
        for record in batch['records']:
            unit = units.get(record['unit_id'])
            if unit is None or record['document_id'] != stem or record['unit_id'] in seen:
                raise ValueError('Stale or duplicate FDD embedding unit')
            seen.add(record['unit_id'])
            expected = unit.get('retrieval_text') or unit['text']
            if (record['text'] != expected or record.get('source_text') != unit['text'] or
                record['content_hash'] != compute_content_hash(expected) or
                record['embedding_model'] != code.embedding_model or
                record['artifact_version'] != stage['embedding_record_artifact_version'] or
                record['cache_key'] != compute_embedding_cache_key(record['content_hash'],
                    record['embedding_model'], record['artifact_version'])):
                raise ValueError('FDD embedding content/model binding mismatch')
            if record['embedding_status'] in {'embedded', 'cached'} and record['vector']:
                usable.append(record)
        if len(usable) != len(units):
            diagnostics.append(dict(document_id=stem, kind='incomplete_embedding_coverage',
                                    units=len(units), usable=len(usable)))
        if not usable:
            documents.append(dict(document_id=stem, status='missing_embeddings', candidates=[]))
            continue
        # Memory bounded per document and 128 FDD units; no vector-store process or network.
        best_scores = np.full(len(code.records), -2.0)
        best_fdd = np.zeros(len(code.records), dtype=int)
        for begin in range(0, len(usable), 128):
            matrix = unit_vectors([r['vector'] for r in usable[begin:begin+128]], code.vector_dimension)
            scores = matrix @ code_vectors.T
            rows = scores.argmax(axis=0)
            maxima = scores[rows, np.arange(len(code.records))]
            improve = maxima > best_scores
            best_scores[improve] = maxima[improve]
            best_fdd[improve] = begin + rows[improve]
        ranked = []
        for symbol_index, indexes in record_groups.items():
            symbol = implementations[symbol_index]
            matches = []
            for region in regions_by_path[symbol.source_path]:
                documents_for_region = scoped_registry.get(
                    (region.identity, symbol.source_path.replace('\\', '/')),
                    registry.get(region.identity, set()),
                )
                if (
                    region.start_offset < symbol.source_map.end_offset
                    and symbol.source_map.start_offset < region.end_offset
                    and documents_for_region == {stem}
                ):
                    matches.append(region)
            # For comment candidates, select evidence from inside the marked block when possible.
            marked_indexes = [i for i in indexes if any(r.start_offset < code.records[i].source_map.end_offset
                              and code.records[i].source_map.start_offset < r.end_offset for r in matches)]
            winner = max(marked_indexes or indexes, key=lambda i: (float(best_scores[i]), code.records[i].unit_id))
            score = float(best_scores[winner])
            if not matches and score < minimum_similarity:
                continue
            record = code.records[winner]
            fdd = usable[int(best_fdd[winner])]
            companions = [s for s in symbols if s.occurrence_role == 'declaration' and
                          s.canonical_qualified_name == symbol.canonical_qualified_name and
                          s.overload_discriminator_hash == symbol.overload_discriminator_hash]
            target = dict(module_id=code.module_id, path=symbol.source_path,
                          qualified_name=symbol.canonical_qualified_name, symbol_kind=symbol.symbol_kind,
                          overload_discriminator_hash=symbol.overload_discriminator_hash, selector_scope='overload')
            ranked.append(dict(status='candidate', basis='comment_region_and_similarity' if matches else 'similarity_only',
                cosine_similarity=round(score, 6), target=target, symbol_occurrence_id=symbol.occurrence_id,
                symbol_source_map=symbol.source_map.model_dump(), code_unit_id=record.unit_id,
                code_source_map=record.source_map.model_dump(), code_excerpt=record.citation_text[:1000],
                fdd_unit_id=fdd['unit_id'], fdd_excerpt=fdd['source_text'][:1000],
                fdd_source_kind=fdd['source_kind'], fdd_source_range=fdd.get('source_range'),
                comment_regions=[r.as_dict() for r in matches],
                declaration_companions=[dict(path=s.source_path, qualified_name=s.canonical_qualified_name,
                    overload_discriminator_hash=s.overload_discriminator_hash, source_map=s.source_map.model_dump()) for s in companions]))
        ranked.sort(key=lambda row: (-bool(row['comment_regions']), -row['cosine_similarity'],
                                    row['target']['path'], row['symbol_occurrence_id']))
        for row in ranked:
            row['proposal_id'] = hashlib.sha256(json.dumps(dict(fdd=stem, target=row['target'],
                snapshot=code.snapshot_id, generation=stage['index_generation']), sort_keys=True).encode()).hexdigest()
        ambiguous = len(ranked) > 1 and ranked[0]['basis'] == ranked[1]['basis'] and abs(
            ranked[0]['cosine_similarity'] - ranked[1]['cosine_similarity']) < ambiguity_margin
        documents.append(dict(document_id=stem, status='candidates_for_review' if ranked else 'no_strong_candidate',
            ambiguous_top_match=ambiguous, total_candidates=len(ranked), returned_candidates=min(top_k, len(ranked)),
            truncated=len(ranked) > top_k, candidates=ranked[:top_k]))
    unmapped = sorted(
        {r['identity'] for f in marker_report for r in f['regions']}
        - registered_identities
    )
    output = dict(schema_version=SCHEMA, status='candidate', external_api_calls=0, automatic_approvals=0,
        fdd_generation=stage['index_generation'], code_snapshot_id=code.snapshot_id,
        parse_generation=code.parse_generation, embedding_model=code.embedding_model,
        vector_dimension=code.vector_dimension, code_file_count=len(entries), code_unit_count=len(code.records),
        comparable_routines=len(record_groups), unassigned_code_units=len(code.records)-sum(map(len, record_groups.values())),
        top_k=top_k, minimum_similarity=minimum_similarity, ambiguity_margin=ambiguity_margin,
        scoring='maximum chunk-pair cosine per implementation routine; exact registered comment regions ranked first',
        scores_are_probabilities=False, thresholds_calibrated=False, input_sha256=bindings,
        documents=documents, comment_inventory=marker_report, unmapped_enhancement_identities=unmapped,
        diagnostics=diagnostics)
    output['report_identity_sha256'] = hashlib.sha256(json.dumps(output, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return output


def render_report(report: dict) -> str:
    def esc(value):
        return html.escape(str(value), quote=True)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>FDD / code candidate review</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172331}',
        'input{padding:12px;width:90%;margin:16px 0}details{border:1px solid #ccd6df;padding:16px;margin:12px 0}',
        'pre{white-space:pre-wrap;background:#f3f6f8;padding:12px;overflow-wrap:anywhere}',
        'summary{cursor:pointer;overflow-wrap:anywhere}article{border-top:1px solid #ddd;padding:12px 0}</style>',
        '<h1>FDD / code candidate review</h1><p>All suggestions are unapproved. Similarity is not proof of implementation. '
        'Read the FDD passage, marked code region and routine together. Full ranges are in the JSON report.</p>',
        f'<p>{esc(report["fdd_generation"])} · {esc(report["code_snapshot_id"])} · '
        f'{report["code_file_count"]} code files · {len(report["documents"])} documents · zero external API calls</p>',
        '<p>A missing match may mean missing source, poor similarity, or incomplete comment correspondence. '
        'Close scores can represent multiple valid routines. Thresholds are exploratory, not calibrated.</p>',
        '<input id="filter" placeholder="Filter by FDD, routine, package, marker or status" aria-label="Filter documents">']
    for doc in report['documents']:
        parts.append(f'<details class="doc"><summary>{esc(doc["document_id"])} — {esc(doc["status"])}'
                     f' ({len(doc["candidates"])}/{doc.get("total_candidates",0)} candidates)</summary>')
        if doc.get('ambiguous_top_match'):
            parts.append('<p>Close top scores: review alternatives; no unique match established.</p>')
        for row in doc['candidates']:
            target = row['target']
            parts.append(f'<article><h3>{esc(target["qualified_name"])}</h3><p>{esc(target["path"])} · '
                         f'{esc(row["basis"])} · cosine {row["cosine_similarity"]:.4f}</p>')
            parts.append(f'<p>Proposal: <code>{esc(row["proposal_id"])}</code></p>')
            for region in row['comment_regions']:
                parts.append(f'<p>Comment: {esc(region["label"])} — lines {region["start_line"]}–{region["end_line"]}</p>')
            parts.append(f'<h4>FDD excerpt</h4><pre>{esc(row["fdd_excerpt"])}</pre>')
            parts.append(f'<h4>Code excerpt (unit lines {row["code_source_map"]["start_line"]}–'
                         f'{row["code_source_map"]["end_line"]})</h4><pre>{esc(row["code_excerpt"])}</pre>')
            parts.append('<p>Review: Does this code implement or support this FDD requirement? '
                         'If uncertain, leave it unresolved.</p></article>')
        parts.append('</details>')
    parts.append('<script>document.getElementById("filter").addEventListener("input",function(){'
                 'const q=this.value.toLowerCase();document.querySelectorAll(".doc").forEach('
                 'x=>x.hidden=!x.textContent.toLowerCase().includes(q));});</script></html>')
    return '\n'.join(parts)


def write_report(report: dict, output_directory: Path) -> None:
    # mkdir(exist_ok=False) also refuses concurrent writers. Partial runs stay visible.
    output_directory.mkdir(parents=True, exist_ok=False)
    with (output_directory / 'proposals.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    with (output_directory / 'review.html').open('x', encoding='utf-8') as stream:
        stream.write(render_report(report))
