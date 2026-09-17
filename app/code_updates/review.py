"""One evidence-bound packet, with explicit human decisions and no auto approval."""
from __future__ import annotations

import html
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from app.code_updates.storage import atomic_text, digest, read
from app.code_ingestion.dependency_review_ledger import (
    DependencyReviewLedger, DependencyReviewDecision, _ledger_identity,
)
from app.fdd_code_lineage.evaluation import CodeCombinedEvalCase
from app.fdd_code_lineage.models import FddCodeTarget, create_mapping, build_lineage_artifact


def item(kind, evidence, question, provenance="machine_proposal"):
    return {"id": digest({"kind": kind, "evidence": evidence}), "kind": kind,
            "evidence": evidence, "question": question, "provenance": provenance}


def render(items, path):
    identity = digest(items)
    text = ("# Consolidated code update review\n\n"
            f"Evidence SHA-256: `{identity}`\n\n"
            "Edit Decision, Rationale and Correction JSON only. Decision: accepted, corrected, "
            "deferred (new lineage only), or needs_more_context. Leave pending until reviewed.\n"
            "Corrections are a JSON object merged into the proposed dependency, target or evaluation; "
            "use {} when accepting. Similarity is a candidate, not proof.\n")
    for entry in items:
        text += (f"\n## {entry['id']}\n\nKind: {entry['kind']}\n\n{entry['question']}\n\n"
                 f"Provenance: {entry['provenance']}\n\n"
                 "```json\n" + json.dumps(entry["evidence"], indent=2, ensure_ascii=False) + "\n```\n\n"
                 + ("Decision: carried\nRationale: carried from bound prior review\nCorrection JSON: {}\n"
                  if entry['provenance'] == 'validated_carry_forward' else "Decision: pending\nRationale: \nCorrection JSON: {}\n"))
    atomic_text(path, text)
    atomic_text(path.with_suffix(".html"), '<!doctype html><meta charset="utf-8"><title>SME review</title>'
                '<h1>Read-only review preview</h1><p>Edit the adjacent review.md file.</p>'
                '<pre style="white-space:pre-wrap">' + html.escape(text) + '</pre>')


def decisions(items, path):
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    header = re.findall(r"^Evidence SHA-256: `([0-9a-f]{64})`$", text, re.M)
    if header != [digest(items)]:
        raise ValueError("Review packet is bound to different evidence")
    sections = re.findall(r"^## ([0-9a-f]{64})\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if len(sections) != len(items) or len({key for key, _ in sections}) != len(items):
        raise ValueError("Review must contain every item exactly once")
    found = dict(sections)
    result = {}
    for entry in items:
        if entry["id"] not in found:
            raise ValueError("Review item identity changed")
        body = found[entry["id"]]
        evidence = re.findall(r"```json\n(.*?)\n```", body, re.S)
        if len(evidence) != 1 or json.loads(evidence[0]) != entry["evidence"]:
            raise ValueError("Read-only review evidence was edited; use Correction JSON")
        def field(name):
            values = re.findall(r"^" + name + r":[ \t]*(.*)$", body, re.M)
            if len(values) != 1:
                raise ValueError(f"Missing or duplicate {name} in {entry['id']}")
            return values[0].strip()
        verdict, rationale = field("Decision"), field("Rationale")
        correction = json.loads(field("Correction JSON"))
        if entry["provenance"] == "validated_carry_forward":
            if verdict != "carried" or correction:
                raise ValueError("Carried decisions are bound to prior evidence; changed decisions require a fresh review item")
            result[entry["id"]] = entry["evidence"]["carry_decision"]
            continue
        if verdict not in {"accepted", "corrected", "deferred"} or not rationale:
            raise ValueError(f"Unresolved SME item {entry['id']}; supply decision and rationale")
        if entry["kind"] in {"lineage", "inherited_lineage"} and verdict != "deferred" and len(rationale) < 10:
            raise ValueError(f"SME item {entry['id']}: lineage rationale requires at least 10 characters explaining the evidence")
        if verdict == "deferred" and entry["kind"] != "lineage":
            raise ValueError("Only new lineage candidates can be deferred")
        if not isinstance(correction, dict) or (verdict != "corrected" and correction):
            raise ValueError("Only corrected decisions may supply a JSON correction")
        result[entry["id"]] = {"verdict": verdict, "rationale": rationale, "correction": correction}
    return result


def collect_review_decisions(primary_items, primary_path, *, amendment_items=(), amendment_path=None):
    """Load a reviewed packet plus an optional, separately bound amendment.

    An amendment is deliberately a second packet rather than a rewrite of the
    original consolidated review. That retains the original evidence binding
    and decisions while allowing a regression-discovered candidate to receive
    one narrow SME decision.
    """
    selected = decisions(primary_items, primary_path)
    all_items = list(primary_items)
    if amendment_items:
        if amendment_path is None:
            raise ValueError("Amendment evidence requires an amendment review packet")
        amendment_selected = decisions(amendment_items, amendment_path)
        duplicate_ids = set(selected).intersection(amendment_selected)
        if duplicate_ids:
            raise ValueError("Review amendment duplicates an existing review item")
        selected.update(amendment_selected)
        all_items.extend(amendment_items)
    return all_items, selected


def combined_review_identity(primary_path, amendment_path=None):
    """Return the hash-bound identity of one or two independently reviewed packets."""
    payload = {"primary_review_sha256": sha_file(primary_path)}
    if amendment_path is not None:
        payload["amendment_review_sha256"] = sha_file(amendment_path)
    return digest(payload)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_ledger(packet, packet_path, review_hash, items, selected, reviewer):
    values = []
    lookup = {i["evidence"]["review_id"]: i for i in items if i["kind"] == "dependency"}
    for case in packet.cases:
        entry = lookup[case.review_id]
        decision = selected[entry["id"]]
        correction = decision["correction"]
        if set(correction) - {"dependency_kind", "resolution_state"}:
            raise ValueError("Dependency corrections allow dependency_kind and resolution_state only")
        from app.code_ingestion.code_analysis_models import DependencyEdge
        import typing
        kind = correction.get("dependency_kind", case.proposed_dependency_kind)
        state = correction.get("resolution_state", case.proposed_resolution_state)
        if kind not in typing.get_args(DependencyEdge.model_fields["dependency_kind"].annotation) or state not in typing.get_args(DependencyEdge.model_fields["resolution_state"].annotation):
            raise ValueError("Invalid corrected dependency kind/state")
        values.append(DependencyReviewDecision(review_id=case.review_id,
            target_canonical_name=case.target_canonical_name, verdict=decision["verdict"],
            effective_dependency_kind=kind, effective_resolution_state=state, rationale=decision["rationale"]))
    from app.code_updates.storage import sha
    payload = dict(schema_version="code_dependency_review_ledger_v1", status="reviewed", reviewer=reviewer,
        snapshot_id=packet.snapshot_id, parser_generation=packet.parser_generation,
        analysis_policy_sha256=packet.analysis_policy_sha256, packet_identity_sha256=packet.packet_identity_sha256,
        packet_json_sha256=sha(packet_path), reviewed_markdown_sha256=review_hash,
        decisions=[v.model_dump(mode="json") for v in values], external_calls_performed=False)
    provisional = DependencyReviewLedger.model_construct(**{**payload, "decisions": tuple(values)}, ledger_identity_sha256="0" * 64)
    return DependencyReviewLedger.model_validate({**payload, "ledger_identity_sha256": _ledger_identity(provisional)})


def reviewed_cases(items, selected):
    result = []
    for entry in items:
        if entry["kind"] != "evaluation":
            continue
        decision = selected[entry["id"]]
        payload = {**entry["evidence"]["case"], **decision["correction"], "sme_reviewed": True,
                   "review_status": "reviewed", "rationale": decision["rationale"]}
        if payload["case_id"] != entry["evidence"]["case"]["case_id"] or payload["mode"] != "code":
            raise ValueError("Evaluation corrections cannot change case identity or mode")
        result.append(CodeCombinedEvalCase.model_validate(payload))
    return result


def normalize_lineage_targets(items, selected, artifact):
    """Rebind an accepted moved source path only when its basename is unique.

    The review packet remains immutable evidence of what the SME saw.  This
    narrow normalization accounts for an intentional directory move between a
    baseline and the current immutable snapshot; it never guesses between two
    same-named files and leaves explicit SME corrections untouched.
    """
    known_paths = {record.source_path for record in artifact.records}
    by_name = {}
    for path in known_paths:
        name = PurePosixPath(path.replace("\\", "/")).name.casefold()
        by_name.setdefault(name, []).append(path)
    overrides, rebindings = {}, []
    for entry in items:
        if entry["kind"] not in {"lineage", "inherited_lineage"}:
            continue
        decision = selected[entry["id"]]
        if decision["verdict"] == "deferred":
            continue
        targets = decision["correction"].get("targets", entry["evidence"]["targets"])
        normalized = []
        for target in targets:
            source_path = target["path"]
            if source_path in known_paths:
                normalized.append(target)
                continue
            if "targets" in decision["correction"]:
                raise ValueError(f"Unknown explicitly corrected code path: {source_path}. Use an exact current snapshot path")
            name = PurePosixPath(source_path.replace("\\", "/")).name.casefold()
            matches = sorted(by_name.get(name, ()))
            if len(matches) != 1:
                choices = ", ".join(matches) if matches else "none"
                raise ValueError(
                    f"Unknown code path: {source_path}. Cannot safely rebind by filename; "
                    f"current matches: {choices}"
                )
            rebound = matches[0]
            normalized.append({**target, "path": rebound})
            rebindings.append({
                "review_item_id": entry["id"],
                "from_path": source_path,
                "to_path": rebound,
                "reason": "unique_basename_rebind_after_source_directory_move",
            })
        overrides[entry["id"]] = normalized
    return overrides, rebindings


def finalize_lineage(items, selected, artifact, generation, review_hash, reviewer, *, target_overrides=None):
    candidates, mappings, deferred = [], [], []
    target_overrides = target_overrides or {}
    for entry in items:
        if entry["kind"] not in {"lineage", "inherited_lineage"}:
            continue
        decision = selected[entry["id"]]
        if decision["verdict"] == "deferred":
            deferred.append({**entry, "decision": decision})
            continue
        data = entry["evidence"]
        targets = target_overrides.get(entry["id"], decision["correction"].get("targets", data["targets"]))
        if set(decision["correction"]) - {"targets", "fdd_document_id", "fdd_release_label", "evaluation_question"}:
            raise ValueError("Unsupported lineage correction")
        params = dict(fdd_document_id=decision["correction"].get("fdd_document_id", data["fdd_document_id"]),
            fdd_release_label=decision["correction"].get("fdd_release_label", data["fdd_release_label"]),
            code_snapshot_id=artifact.snapshot_id, targets=[FddCodeTarget.model_validate(t) for t in targets],
            rationale=decision["rationale"])
        candidates.append(create_mapping(**params))
        mapping_reviewer = decision.get("prior_reviewer", reviewer)
        mappings.append(create_mapping(**params, mapping_status="reviewed", reviewer=mapping_reviewer))
    candidate = build_lineage_artifact(fdd_generation=generation, code_artifact=artifact, mappings=candidates)
    reviewed = build_lineage_artifact(fdd_generation=generation, code_artifact=artifact, mappings=mappings,
        source_candidate_artifact_identity_sha256=candidate.artifact_identity_sha256,
        review_packet_sha256=review_hash, reviewer=reviewer)
    return candidate, reviewed, deferred
