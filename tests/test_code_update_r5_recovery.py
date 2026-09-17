"""Recurring-run regressions discovered in the R4 operator audit."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.code_updates import review
from app.code_updates.coordinator import Coordinator, finalized_review_identity
from app.code_updates.storage import Run, read, write


def test_rebinding_tree_with_different_order_preserves_original_binding(tmp_path):
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "a.sql").write_text("a")
    (tree / "B.sql").write_text("b")
    run = Run(tmp_path, "R5")
    run.bind_tree(tree)
    bound = run.state["trees"][str(tree.resolve())]
    bound["members"].reverse()
    original = list(bound["members"])
    run.bind_tree(tree)
    run.check_bindings()
    assert bound["members"] == original
    (tree / "extra.sql").write_text("extra")
    with pytest.raises(ValueError, match="membership changed"):
        run.bind_tree(tree)


def test_lineage_short_rationale_fails_with_item_id(tmp_path):
    entry = review.item("lineage", {}, "Is this supported?")
    path = tmp_path / "review.md"
    review.render([entry], path)
    path.write_text(path.read_text().replace("Decision: pending", "Decision: accepted")
                    .replace("Rationale: \n", "Rationale: accepted\n"), encoding="utf-8")
    with pytest.raises(ValueError, match=entry["id"] + ": lineage rationale"):
        review.decisions([entry], path)


def test_explicit_corrected_path_is_not_silently_rebound(tmp_path):
    entry = review.item("lineage", {"targets": [{"path": "old/pkg.sql"}]}, "Check target")
    selected = {entry["id"]: {"verdict": "corrected", "correction": {
        "targets": [{"path": "mistyped/pkg.sql"}]}}}
    artifact = SimpleNamespace(records=[SimpleNamespace(source_path="new/pkg.sql")])
    with pytest.raises(ValueError, match="explicitly corrected code path"):
        review.normalize_lineage_targets([entry], selected, artifact)


@pytest.mark.parametrize("amended", [False, True])
def test_reviewed_embedding_resume_uses_finalization_identity(tmp_path, monkeypatch, amended):
    run = Coordinator(tmp_path, "R5", prompt=lambda _: pytest.fail("Already approved"))
    run.dir.mkdir(parents=True)
    run.path("review.md").write_text("primary review", encoding="utf-8")
    if amended:
        run.path("review_amendment.md").write_text("amended review", encoding="utf-8")
        write(run.path("review_amendment_evidence.json"), [{"id": "amendment"}])
    plan = run.path("extra-request.json")
    approval = run.path("extra-approval.json")
    output = run.path("extra-embedded.json")
    write(plan, {"request_hash": "planned"})
    write(approval, {"request_hash": "planned", "max_usd": "1"})
    pending = dict(review_hash=finalized_review_identity(run.dir), request=str(plan),
                   approval=str(approval), prepared="prepared", batches=str(run.path("batches")), output=str(output))
    run.run.state["pending_review_embedding"] = dict(pending)
    monkeypatch.setattr("app.code_updates.coordinator.load_code_index_artifact", lambda _: "prepared")
    calls = []
    def embed(*args):
        calls.append(args)
        return SimpleNamespace(model_dump=lambda **_: {"cached": True})
    monkeypatch.setattr("app.code_updates.coordinator.checkpointed_embed", embed)
    run.build(SimpleNamespace())
    assert len(calls) == 1 and read(output) == {"cached": True}
    assert run.run.state["next_action"] == "finalize"
    run.run.state["pending_review_embedding"] = pending
    changed = run.path("review_amendment.md" if amended else "review.md")
    changed.write_text("different decisions", encoding="utf-8")
    with pytest.raises(ValueError, match="Review changed"):
        run.build(SimpleNamespace())
    assert len(calls) == 1


def test_threshold_pass_with_failed_case_is_not_checkpointed(tmp_path, monkeypatch):
    run = Coordinator(tmp_path, "R5")
    run.dir.mkdir(parents=True)
    run.run.state["analysis"] = "analysis"
    artifact = SimpleNamespace(artifact_identity_sha256="artifact", snapshot_id="snapshot")
    monkeypatch.setattr("app.code_updates.coordinator.load_code_index_artifact", lambda _: artifact)
    failed = True
    def command(script, *args, **kwargs):
        write(Path(args[-1]), {"metadata": {"reviewed_manifest": True,
              "code_artifact_identity_sha256": "artifact", "code_snapshot_id": "snapshot", "eval_file_sha256": {}},
              "summary": {"release_gate_eligible": True},
              "cases": [{"mode": "code", "failures": ["missing routine"] if failed else []}]})
        return 0
    monkeypatch.setattr(run, "command", command)
    with pytest.raises(RuntimeError, match="attempt1.*failed cases"):
        run.evaluate("artifact", None, ["cases"], [], "final-test")
    assert not run.path("final-test_code_evaluation.json").exists()
    failed = False
    run.evaluate("artifact", None, ["cases"], [], "final-test")
    assert run.path("final-test_code_evaluation_attempt1.json").exists()
    assert run.path("final-test_code_evaluation_attempt2.json").exists()
    assert read(run.path("final-test_code_evaluation.json"))["cases"][0]["failures"] == []


def test_amendment_needs_business_query_and_rejects_activated_run(tmp_path):
    run = Coordinator(tmp_path, "R5")
    run.dir.mkdir(parents=True)
    run.run.state["steps"]["review_packet"] = {}
    args = SimpleNamespace(fdd_document_id="R25", source_path="SQL/pkg.sql", qualified_name="PKG.PROCESS",
                           symbol_kind="procedure", source_marker="enhancement")
    with pytest.raises(ValueError, match="requires: evidence_query"):
        run.amend_review(args)
    write(run.path("activation.json"), {"applied": True})
    with pytest.raises(ValueError, match="was activated"):
        run.amend_review(args)


def test_next_run_init_inherits_passed_boundary_report(tmp_path, monkeypatch):
    from app.code_updates.storage import sha
    run = Coordinator(tmp_path, "R5", prompt=lambda _: pytest.fail("All inputs supplied"))
    run.dir.mkdir(parents=True)
    settings = SimpleNamespace(code_index_artifact_path=tmp_path / "embedded.json",
        code_snapshots_dir=tmp_path / "snapshots", code_analysis_directory=tmp_path / "analysis",
        fdd_code_lineage_artifact_path=tmp_path / "lineage.json", fdd_generation="v9",
        fdd_retrieval_artifact_dir=tmp_path / "fdd", data_dir=tmp_path / "data",
        code_qdrant_local_path=tmp_path / "store")
    artifact = SimpleNamespace(snapshot_id="fci-custom-r4-abcdef", dependency_review_ledger_sha256="ledger",
        parse_generation="parser", embedding_model="text-embedding-3-large")
    monkeypatch.setattr("app.code_updates.coordinator.Settings", lambda: settings)
    monkeypatch.setattr("app.code_updates.coordinator.load_code_index_artifact", lambda _: artifact)
    monkeypatch.setattr("app.code_updates.coordinator.load_snapshot_manifest",
                        lambda _: SimpleNamespace(request=SimpleNamespace(module_set="fci-custom")))
    evidence = {}
    for mode in ("code", "combined"):
        manifest = tmp_path / f"{mode}.jsonl"
        manifest.write_text("{}\n")
        report = tmp_path / f"{mode}.json"
        write(report, {"metadata": {"eval_file_sha256": {str(manifest): sha(manifest)}},
                       "summary": {"release_gate_eligible": True}, "cases": [{"mode": mode}]})
        evidence[report.name] = sha(report)
    boundary = tmp_path / "boundary.jsonl"
    boundary.write_text("{}\n")
    boundary_report = tmp_path / "boundary.json"
    write(boundary_report, {"schema_version": "code_documentation_boundary_eval_v1",
        "metadata": {"eval_file": str(boundary), "eval_file_sha256": sha(boundary)},
        "summary": {"passed": True}, "cases": [{}]})
    evidence[boundary_report.name] = sha(boundary_report)
    ledger = tmp_path / "ledger.json"
    write(ledger, {"ledger_identity_sha256": "ledger"})
    evidence[ledger.name] = sha(ledger)
    monkeypatch.setattr("app.code_updates.coordinator.active_baseline",
                        lambda *_: (tmp_path / "promotion.json", {"evidence_files": evidence}))
    monkeypatch.setattr(run, "complete_init", lambda: None)
    run.init(SimpleNamespace(source_directory=str(tmp_path), svn_revision="5", reviewer="SME",
        application_build="14.7", price_per_million="0.13", pricing_basis="fixture"))
    assert run.config["boundary_evals"] == [str(boundary)]
    assert run.config["request"]["base_snapshot_id"] == artifact.snapshot_id


def test_amendment_uses_operator_query_for_non_aml_requirement(tmp_path, monkeypatch):
    run = Coordinator(tmp_path, "R5")
    run.dir.mkdir(parents=True)
    run.run.state.update(steps={"review_packet": {}}, analysis="analysis")
    run.config = {"fdd_directory": "fdd", "snapshot_root": str(tmp_path / "snapshots")}
    source = tmp_path / "snapshots/current/source/SQL/pkg_claim.sql"
    source.parent.mkdir(parents=True)
    source.write_text("-- R25 claim reversal\nPROCEDURE reverse_claim IS BEGIN NULL; END;", encoding="utf-8")
    location = SimpleNamespace(start_offset=0, end_offset=100,
        model_dump=lambda **_: {"source_path": "SQL/pkg_claim.sql", "start_line": 1, "end_line": 2})
    symbol = SimpleNamespace(canonical_qualified_name="PKG_CLAIM.REVERSE_CLAIM", symbol_kind="procedure",
                             source_map=location)
    artifact = SimpleNamespace(snapshot_id="current", module_id="claims", artifact_identity_sha256="artifact",
        records=[SimpleNamespace(source_path="SQL/pkg_claim.sql", source_map=location, citation_text="reverse_claim")])
    monkeypatch.setattr("app.code_updates.coordinator.load_code_index_artifact", lambda _: artifact)
    monkeypatch.setattr("app.fdd_code_lineage.models._load_analysis", lambda _: {"SQL/pkg_claim.sql": [symbol]})
    document = SimpleNamespace(document_id="FS_R25_Claims_v1")
    monkeypatch.setattr("app.retrieval.lexical_search.load_retrieval_ready_documents", lambda _: [document])
    query = "same-day claim reversal validation"
    searches = []
    def search(documents, supplied_query, *, limit):
        searches.append((documents, supplied_query, limit))
        return [SimpleNamespace(point_id="claim-passage", payload={"text": query})]
    monkeypatch.setattr("app.retrieval.lexical_search.search_lexical_documents", search)
    run.amend_review(SimpleNamespace(fdd_document_id=document.document_id, source_path="SQL/pkg_claim.sql",
        qualified_name=symbol.canonical_qualified_name, symbol_kind="procedure", source_marker="R25 claim reversal",
        evidence_query=query))
    entry = read(run.path("review_amendment_evidence.json"))[0]
    assert searches == [([document], query, 2)]
    assert query in entry["evidence"]["evaluation_question"]
    assert entry["evidence"]["proposal"]["fdd_passages"][0]["unit_id"] == "claim-passage"
    assert "offline" not in entry["question"]
