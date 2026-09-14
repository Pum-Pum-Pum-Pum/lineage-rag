"""Conservative carry-forward: changed or unknown caller context prevents reuse."""
from __future__ import annotations

from pathlib import Path
from collections import Counter

from app.code_updates.storage import digest, read


def source_fingerprints(artifact, source_path):
    return Counter((r.content_sha256, r.cache_key, r.source_kind, r.display_name,
                    r.source_map.start_line, r.source_map.end_line,
                    r.source_map.start_offset, r.source_map.end_offset,
                    r.parent_source_map.start_line if r.parent_source_map else None,
                    r.parent_source_map.end_line if r.parent_source_map else None)
                   for r in artifact.records if r.source_path == source_path)


def affected_context(base_analysis, target_analysis, changed_paths):
    graph = {}
    unknown = set()
    for directory in (base_analysis, target_analysis):
        files = [read(p) for p in (Path(directory) / "analysis").glob("*.json")]
        names = {}
        ids = {}
        for file in files:
            path = file["source_path"]
            graph.setdefault(path, set())
            for symbol in file.get("symbols", []):
                ids[symbol["occurrence_id"]] = path
                full = symbol["canonical_qualified_name"].casefold()
                for name in (full, full.rsplit(".", 1)[0]):
                    names.setdefault(name, set()).add(path)
        for file in files:
            for edge in file.get("dependencies", []):
                source = file["source_path"]
                targets = {ids[i] for i in edge.get("candidate_symbol_occurrence_ids", []) if i in ids}
                name = edge["target_canonical_name"].casefold()
                targets.update(names.get(name, set()))
                if "." in name:
                    targets.update(names.get(name.rsplit(".", 1)[0], set()))
                for target in targets:
                    graph.setdefault(source, set()).add(target)
                    graph.setdefault(target, set()).add(source)
                if edge["dependency_kind"] in {"routine_call", "dynamic_sql", "external_package"} and not targets:
                    unknown.add(source)
    affected = set(changed_paths)
    # A modified dynamic/unresolved caller may reach any previously mapped code.
    if affected & unknown:
        return set(graph)
    pending = list(affected)
    while pending:
        for neighbor in graph.get(pending.pop(), ()):
            if neighbor not in affected:
                affected.add(neighbor)
                pending.append(neighbor)
    return affected


def case_fingerprint(case):
    payload = case.model_dump(mode="json") if hasattr(case, "model_dump") else dict(case)
    for name in ("review_id", "sme_verdict", "sme_rationale"):
        payload.pop(name, None)
    payload["examples"] = [{k: v for k, v in example.items() if k != "source_symbol_occurrence_id"}
                           for example in payload.get("examples", [])]
    return digest(payload)


def carried_dependency_decisions(base_packet, target_packet, base_ledger, affected):
    if base_packet.analysis_policy_sha256 != target_packet.analysis_policy_sha256:
        return {}
    prior = {case_fingerprint(c): c for c in base_packet.cases}
    decisions = {d.review_id: d for d in base_ledger.decisions}
    carried = {}
    for case in target_packet.cases:
        old = prior.get(case_fingerprint(case))
        if old is None or old.review_id not in decisions or any(e.source_path in affected for e in case.examples):
            continue
        decision = decisions[old.review_id]
        carried[case.review_id] = {"verdict": decision.verdict, "rationale": decision.rationale,
            "correction": {"dependency_kind": decision.effective_dependency_kind,
                           "resolution_state": decision.effective_resolution_state} if decision.verdict == "corrected" else {},
            "prior_review_id": old.review_id, "prior_reviewer": base_ledger.reviewer,
            "prior_ledger_identity": base_ledger.ledger_identity_sha256}
    return carried
