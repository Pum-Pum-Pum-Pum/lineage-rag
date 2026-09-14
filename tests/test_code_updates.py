from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.code_updates.storage import Run, locked, read, write, digest, sha
from app.code_updates.embedding import plan_embeddings, checkpointed_embed
from app.code_updates import review
from app.code_updates.inheritance import affected_context
from app.code_retrieval.service import retrieve_code_evidence
from app.code_indexing.contract import write_code_index_artifact_no_overwrite
from app.code_indexing.embedding import embed_code_index_artifact
from test_code_indexing import _prepared


class Provider:
    def __init__(self, fail_on=None):
        self.embeddings = self
        self.calls = 0
        self.fail_on = fail_on

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on:
            raise TimeoutError("response lost")
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.5] * 3072)
                                      for i in reversed(range(len(kwargs["input"])))])


def planned(tmp_path):
    artifact = _prepared(tmp_path)
    plan = plan_embeddings(artifact, [], [], "0.13", "operator confirmed test price")
    plan["batch_size"] = 1
    plan["request_hash"] = digest({k: v for k, v in plan.items() if k != "request_hash"})
    approval = {"request_hash": plan["request_hash"], "decision": "approved", "max_usd": "1"}
    return artifact, plan, approval


def test_preparation_has_no_provider_and_frozen_input_counts(tmp_path):
    artifact, plan, approval = planned(tmp_path)
    assert plan["unique_missing_inputs"] == 2
    assert plan["token_count_method"] == "cl100k_base"
    assert plan["token_upper_bound"] > 0
    assert artifact.dependency_review_status == "draft"


def test_successful_batches_are_not_resent_after_restart(tmp_path):
    artifact, plan, approval = planned(tmp_path)
    provider = Provider()
    first = checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: provider)
    second = checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: pytest.fail("No new client"))
    assert provider.calls == 2
    assert first == second
    assert second.dependency_review_status == "draft"


def test_uncertain_paid_outcome_holds_resume_without_resending(tmp_path):
    artifact, plan, approval = planned(tmp_path)
    provider = Provider(fail_on=2)
    with pytest.raises(TimeoutError):
        checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: provider)
    assert (tmp_path / "batches/00000000.result.json").is_file()
    with pytest.raises(RuntimeError, match="Uncertain paid"):
        checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: pytest.fail("No resend"))
    assert provider.calls == 2


def test_budget_checks_before_request(tmp_path):
    artifact, plan, approval = planned(tmp_path)
    approval["max_usd"] = "0.000000000001"
    with pytest.raises(PermissionError, match="budget"):
        checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: pytest.fail("No paid request"))
    assert not list((tmp_path / "batches").glob("*.intent.json"))


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1", "0"])
def test_invalid_budgets_rejected(tmp_path, amount):
    artifact, plan, approval = planned(tmp_path)
    approval["max_usd"] = amount
    with pytest.raises(ValueError):
        checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: pytest.fail("No client"))


def test_changed_approval_and_request_rejected(tmp_path):
    artifact, plan, approval = planned(tmp_path)
    approval["request_hash"] = "b" * 64
    with pytest.raises(PermissionError):
        checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: pytest.fail("No client"))
    plan["missing"][0]["token_upper_bound"] = 1
    with pytest.raises(ValueError, match="identity"):
        checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: pytest.fail("No client"))


def test_unchanged_source_cache_miss_stops(tmp_path):
    artifact = _prepared(tmp_path)
    with pytest.raises(ValueError, match="unchanged source"):
        plan_embeddings(artifact, [], ["pkg_claim.sql"], "0.13", "test")


def test_cache_only_and_metadata_rebinding_never_calls_provider(tmp_path):
    artifact, plan, approval = planned(tmp_path)
    base = checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: Provider())
    path = write_code_index_artifact_no_overwrite(base, tmp_path / "embedded")
    cache_plan = plan_embeddings(artifact, [path], ["pkg_claim.sql"], "0.13", "test")
    assert cache_plan["unique_missing_inputs"] == 0
    cache_approval = {"request_hash": cache_plan["request_hash"], "decision": "approved", "max_usd": "1"}
    new = checkpointed_embed(artifact, cache_plan, cache_approval, tmp_path / "cached", lambda: pytest.fail("No calls"))
    assert all(r.embedding_status == "cached" for r in new.records)


def test_provisional_diagnostics_cannot_enter_normal_retrieval(tmp_path):
    artifact, plan, approval = planned(tmp_path)
    embedded = checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: Provider())
    with pytest.raises(ValueError, match="reviewed"):
        retrieve_code_evidence(artifact=embedded, query="process_claim", mode="lexical")
    result = retrieve_code_evidence(artifact=embedded, query="process_claim", mode="lexical", allow_provisional=True)
    assert result.evidence
    with pytest.raises(ValueError, match="lexical only"):
        retrieve_code_evidence(artifact=embedded, query="process_claim", mode="dense", allow_provisional=True)


def test_steps_resume_and_reject_tampered_output(tmp_path):
    run = Run(tmp_path, "R4")
    with locked(run.directory):
        output = run.directory / "artifact.json"
        write(output, {"a": 1})
        assert run.step("one", lambda: ({"result": 1}, [output])) == {"result": 1}
    restarted = Run(tmp_path, "R4")
    assert restarted.step("one", lambda: pytest.fail("Already completed")) == {"result": 1}
    write(output, {"a": 2})
    with pytest.raises(ValueError, match="changed output"):
        restarted.step("one", lambda: pytest.fail("Changed evidence"))


def test_duplicate_coordinators_rejected_and_lock_released(tmp_path):
    with locked(tmp_path):
        with pytest.raises(RuntimeError, match="Another coordinator"):
            with locked(tmp_path):
                pass
    with locked(tmp_path):
        pass


def accept_packet(items, tmp_path, verdict="accepted", correction="{}"):
    path = tmp_path / "review.md"
    review.render(items, path)
    text = path.read_text(encoding="utf-8").replace("Decision: pending", "Decision: " + verdict)
    text = text.replace("Rationale: \n", "Rationale: Checked against actual source evidence.\n")
    text = text.replace("Correction JSON: {}", "Correction JSON: " + correction)
    path.write_text(text, encoding="utf-8")
    return path


def test_deferred_link_remains_candidate_and_no_approval_is_invented(tmp_path):
    artifact = _prepared(tmp_path)
    entry = review.item("lineage", {"fdd_document_id": "R1-doc", "fdd_release_label": "R1",
        "targets": [{"module_id": artifact.module_id, "path": "pkg_claim.sql", "selector_scope": "file",
                     "rationale": "Candidate discovery only."}]}, "Does it implement this requirement?")
    path = accept_packet([entry], tmp_path, "deferred")
    chosen = review.decisions([entry], path)
    candidate, final, deferred = review.finalize_lineage([entry], chosen, artifact, "v9", sha(path), "Pum")
    assert len(deferred) == 1 and not final.mappings
    assert final.status == "reviewed" and candidate.status == "candidate"


def test_dependency_cannot_be_deferred(tmp_path):
    entry = review.item("dependency", {"review_id": "b" * 64}, "Check dependency")
    with pytest.raises(ValueError, match="Only new lineage"):
        review.decisions([entry], accept_packet([entry], tmp_path, "deferred"))


def test_review_rejects_edited_evidence_and_missing_decisions(tmp_path):
    entry = review.item("lineage", {"path": "old.sql"}, "Check link")
    path = tmp_path / "review.md"
    review.render([entry], path)
    with pytest.raises(ValueError, match="Unresolved"):
        review.decisions([entry], path)
    path = accept_packet([entry], tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace('"old.sql"', '"new.sql"'), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence was edited"):
        review.decisions([entry], path)


def test_changed_caller_invalidates_transitive_context(tmp_path):
    base, target = tmp_path / "base", tmp_path / "target"
    for directory in (base, target):
        write(directory / "analysis/a.json", {"source_path": "caller.sql", "symbols": [], "dependencies": [
            {"dependency_kind": "routine_call", "target_canonical_name": "pkg.delete", "candidate_symbol_occurrence_ids": ["id"]}]})
        write(directory / "analysis/b.json", {"source_path": "callee.sql", "symbols": [
            {"occurrence_id": "id", "canonical_qualified_name": "pkg.delete"}], "dependencies": []})
    assert "callee.sql" in affected_context(base, target, ["caller.sql"])


def test_dependency_review_creates_valid_ledger(tmp_path):
    from app.code_ingestion.dependency_review import DependencyReviewPacket, DependencyReviewCase
    case = DependencyReviewCase(review_id="c" * 64, target_canonical_name="TABLE_A",
        proposed_dependency_kind="table_write", proposed_resolution_state="external_schema",
        confidence="high", review_reason="Verify", occurrence_count=1, examples=())
    packet = DependencyReviewPacket(snapshot_id="x-r1-ab", snapshot_content_sha256="a" * 64,
        parser_generation="v1", analysis_policy_sha256="b" * 64, total_review_cases=1,
        total_occurrences=1, packet_identity_sha256="d" * 64, cases=(case,))
    path = tmp_path / "packet.json"
    write(path, packet.model_dump(mode="json"))
    entry = review.item("dependency", case.model_dump(mode="json"), "Check")
    packet_review = accept_packet([entry], tmp_path)
    selected = review.decisions([entry], packet_review)
    ledger = review.dependency_ledger(packet, path, sha(packet_review), [entry], selected, "Pum")
    assert ledger.status == "reviewed" and ledger.decisions[0].effective_dependency_kind == "table_write"


def test_bound_tree_detects_added_file(tmp_path):
    run = Run(tmp_path, "tree")
    run.directory.mkdir(parents=True)
    write(tmp_path / "evidence/a.json", {"a": 1})
    run.bind_tree(tmp_path / "evidence", "*.json")
    write(tmp_path / "evidence/b.json", {"b": 2})
    with pytest.raises(ValueError, match="membership changed"):
        Run(tmp_path, "tree").check_bindings()


def test_baseline_resolution_rejects_ambiguous_applied_requests(tmp_path):
    from app.code_updates.coordinator import active_baseline
    from app.activation.code_generation import digest as promotion_digest
    settings = SimpleNamespace(code_modes_enabled=True, code_index_artifact_path=tmp_path / "code.json",
        code_analysis_directory=tmp_path / "analysis", code_qdrant_collection_name="code_custom_test",
        fdd_code_lineage_artifact_path=tmp_path / "lineage.json")
    target = {"CODE_MODES_ENABLED": "true", "CODE_INDEX_ARTIFACT_PATH": str(settings.code_index_artifact_path),
        "CODE_ANALYSIS_DIRECTORY": str(settings.code_analysis_directory), "CODE_QDRANT_COLLECTION_NAME": settings.code_qdrant_collection_name,
        "FDD_CODE_LINEAGE_ARTIFACT_PATH": str(settings.fdd_code_lineage_artifact_path)}
    for name in ("one", "two"):
        request = {"schema_version": "code_generation_promotion_request_v1", "target_configuration": target, "name": name}
        request["request_identity_sha256"] = promotion_digest(request)
        write(tmp_path / f"data/exports/activation/{name}-request.json", request)
        write(tmp_path / f"data/exports/activation/{name}.result.json", {"action": "activate", "applied": True,
            "request_identity_sha256": request["request_identity_sha256"]})
    with pytest.raises(ValueError, match="found 2"):
        active_baseline(tmp_path, settings)


def test_quota_failure_is_held_and_not_retried(tmp_path):
    artifact, plan, approval = planned(tmp_path)
    class QuotaClient:
        @property
        def embeddings(self):
            return self
        def create(self, **_):
            raise RuntimeError("insufficient_quota")
    with pytest.raises(RuntimeError, match="insufficient_quota"):
        checkpointed_embed(artifact, plan, approval, tmp_path / "batches", QuotaClient)
    with pytest.raises(RuntimeError, match="Uncertain"):
        checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: pytest.fail("must not retry"))


def test_paid_client_rejects_changed_disclosure_endpoint(monkeypatch):
    from app.code_updates.embedding import approved_client
    monkeypatch.setattr("app.core.config.get_settings", lambda: SimpleNamespace(openai_base_url="https://unapproved.example/v1"))
    monkeypatch.setattr("app.embeddings.client.get_embedding_client", lambda **_: pytest.fail("must stop before client"))
    with pytest.raises(PermissionError, match="endpoint"):
        approved_client()


def test_new_source_during_import_is_rejected(tmp_path, monkeypatch):
    from app.code_ingestion import source_import
    source, intake = tmp_path / "source", tmp_path / "intake"
    source.mkdir()
    intake.mkdir()
    (source / "a.sql").write_text("PROCEDURE p IS BEGIN NULL; END;", encoding="utf-8")
    original = source_import.shutil.copyfile
    def changed(src, dest):
        result = original(src, dest)
        (source / "b.sql").write_text("PROCEDURE q IS BEGIN NULL; END;", encoding="utf-8")
        return result
    monkeypatch.setattr(source_import.shutil, "copyfile", changed)
    with pytest.raises(source_import.SourceImportError, match="changed"):
        source_import.stage_external_code_source(source, intake)
    assert not (intake / "source").exists()


def test_budget_extension_resumes_only_unsent_batches(tmp_path):
    from decimal import Decimal
    artifact, plan, approval = planned(tmp_path)
    first_cost = Decimal(plan["missing"][0]["token_upper_bound"]) * Decimal(plan["usd_per_million_tokens"]) / 1000000
    provider = Provider()
    with pytest.raises(PermissionError, match="budget"):
        checkpointed_embed(artifact, plan, {**approval, "max_usd": str(first_cost)}, tmp_path / "batches", lambda: provider)
    assert provider.calls == 1
    checkpointed_embed(artifact, plan, approval, tmp_path / "batches", lambda: provider)
    assert provider.calls == 2


def test_live_restart_receipt_rejects_old_and_tampered_runtime(tmp_path, monkeypatch):
    import hashlib
    import hmac
    from app.code_updates import runtime
    run = Run(tmp_path, "R4")
    run.directory.mkdir(parents=True)
    identity = {"snapshot_id": "r4", "collection": "final"}
    challenge = {"secret": "ab" * 32, "nonce": "unique", "created_at": "2026-09-14T01:00:00+00:00", "expected": identity}
    write(run.directory / "restart_challenge.json", challenge)
    monkeypatch.setattr(runtime, "runtime_identity", lambda _: identity)
    monkeypatch.setattr(runtime, "process_identity", lambda _: (True, "creation"))
    payload = {"run_id": "R4", "nonce": "unique", "identity": identity, "process_id": 123,
               "server_started_at": "2026-09-14T02:00:00+00:00", "process_creation_token": "creation"}
    receipt = tmp_path / "receipt.json"
    def sign(value):
        return {**value, "signature": hmac.new(bytes.fromhex(challenge["secret"]), digest(value).encode(), hashlib.sha256).hexdigest()}
    write(receipt, sign(payload))
    assert runtime.verify_restart_receipt(run, receipt)["passed"]
    write(receipt, sign({**payload, "server_started_at": "2026-09-14T00:00:00+00:00"}))
    with pytest.raises(ValueError, match="not restarted"):
        runtime.verify_restart_receipt(run, receipt)
    write(receipt, {**sign(payload), "identity": {"collection": "old"}})
    with pytest.raises(ValueError, match="not from"):
        runtime.verify_restart_receipt(run, receipt)
    write(receipt, sign(payload))
    monkeypatch.setattr(runtime, "process_identity", lambda _: (False, None))
    with pytest.raises(ValueError, match="no longer running"):
        runtime.verify_restart_receipt(run, receipt)


def test_startup_receipt_is_only_published_for_approved_pending_run(tmp_path, monkeypatch):
    from app.code_updates import runtime
    run = Run(tmp_path, "R4")
    run.directory.mkdir(parents=True)
    run.state["status"] = "AWAITING_RESTART"
    run.save()
    identity = {"snapshot_id": "test"}
    write(run.directory / "restart_challenge.json", {"expected": identity, "secret": "ab" * 32,
        "nonce": "random", "created_at": "2026-09-14T01:00:00+00:00"})
    monkeypatch.setattr(runtime, "runtime_identity", lambda _: identity)
    monkeypatch.setattr(runtime, "process_identity", lambda _: (True, "creation"))
    runtime.publish_restart_receipts(None, "2026-09-14T02:00:00+00:00", root=tmp_path)
    assert not list((run.directory / "runtime_receipts").glob("*.json"))
    write(run.directory / "activation.json", {"applied": True})
    runtime.publish_restart_receipts(None, "2026-09-14T02:00:00+00:00", root=tmp_path)
    receipts = list((run.directory / "runtime_receipts").glob("*.json"))
    assert len(receipts) == 1
    assert runtime.verify_restart_receipt(run, receipts[0])["passed"]


def test_promotion_mutex_rejects_live_owner():
    import os
    import threading
    import uuid
    from app.code_updates.runtime import stopped_mcp_guard
    if os.name != "nt":
        pytest.skip("Windows Desktop ownership guard")
    name = "Local\\CullingBladeUpdateTest" + uuid.uuid4().hex
    ready, release = threading.Event(), threading.Event()
    def owner():
        with stopped_mcp_guard(name):
            ready.set()
            release.wait(10)
    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert ready.wait(3)
        with pytest.raises(RuntimeError, match="still running"):
            with stopped_mcp_guard(name):
                pytest.fail("Should not own a live server mutex")
    finally:
        release.set()
        thread.join(3)
    with stopped_mcp_guard(name):
        pass


def test_collection_collision_preserves_existing_and_resume(tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    from app.code_updates.coordinator import Coordinator
    artifact, plan, approval = planned(tmp_path)
    embedded = checkpointed_embed(artifact, plan, approval, tmp_path / "batches", Provider)
    path = tmp_path / "embedded.json"
    write(path, embedded.model_dump(mode="json"))
    coordinator = Coordinator(tmp_path, "R4-test")
    coordinator.run.directory.mkdir(parents=True)
    coordinator.config = {"code_store": str(tmp_path / "store")}
    name = f"code_custom_update_r4_test_final_{artifact.artifact_identity_sha256[:12]}"
    client = QdrantClient(path=str(tmp_path / "store"))
    client.create_collection(name, vectors_config=VectorParams(size=3072, distance=Distance.COSINE))
    client.close()
    selected = coordinator.stage_collection(path, "final")
    assert selected == name + "_attempt2"
    assert coordinator.stage_collection(path, "final") == selected
    client = QdrantClient(path=str(tmp_path / "store"))
    assert client.count(name).count == 0
    assert client.count(selected).count == 2
    client.close()


def test_partial_init_resumes_request_creation(tmp_path, monkeypatch):
    from app.code_updates.coordinator import Coordinator
    from app.code_ingestion.snapshot_models import SnapshotRequest
    run = Coordinator(tmp_path, "INIT", prompt=lambda _: pytest.fail("resume prompted"))
    request = SnapshotRequest(module_set="fci-custom", svn_revision="154", application_build="14.7",
                              reviewer="SME", base_snapshot_id="fci-custom-r3-abcdef")
    config = {"request": request.model_dump(mode="json"), "source_directory": str(tmp_path),
              "code_evals": [], "combined_evals": [], "boundary_evals": [], "fdd_generation": "v9"}
    for key in ("base_promotion", "base_dependency_ledger", "registry", "base_artifact", "base_lineage",
                "fdd_directory", "base_analysis", "fdd_stage"):
        config[key] = str(tmp_path / key)
    write(run.path("config.json"), config)
    resumed = Coordinator(tmp_path, "INIT")
    monkeypatch.setattr(resumed.run, "bind", lambda _: None)
    monkeypatch.setattr(resumed.run, "bind_tree", lambda *_: None)
    resumed.init(SimpleNamespace())
    assert resumed.run.state["status"] == "INITIALIZED"
    assert read(resumed.path("intake/snapshot_request.json")) == request.model_dump(mode="json")


def test_real_prepare_and_build_resume_without_paid_calls(tmp_path, monkeypatch):
    """Real snapshot/parser/index/Qdrant; fake provider and discovery only."""
    from test_code_parsing_pipeline import _build_snapshot
    from app.code_ingestion.code_parsing_pipeline import parse_code_snapshot, PARSER_GENERATION_DIRECTORY
    from app.code_ingestion.dependency_review import build_dependency_review_packet
    from app.code_indexing.contract import build_code_index_artifact
    from app.code_updates.coordinator import Coordinator
    from app.code_ingestion.snapshot_models import SnapshotRequest
    from app.fdd_code_lineage.models import build_lineage_artifact
    from scripts.check_code_preindex_gate import main as check_parse
    source = "CREATE OR REPLACE PACKAGE BODY pkg_customer_custom AS\nPROCEDURE update_customer IS BEGIN NULL; END;\nEND pkg_customer_custom;\n/\n"
    base_dir, base_manifest = _build_snapshot(tmp_path / "base", source)
    base_staging = tmp_path / "base_parse"
    parse_code_snapshot(base_dir, base_staging)
    analysis = base_staging / base_manifest.snapshot_id / PARSER_GENERATION_DIRECTORY
    base = build_code_index_artifact(analysis, embedding_model="text-embedding-3-large")
    packet = build_dependency_review_packet(base_dir, analysis)
    packet_path = tmp_path / "base_packet.json"
    write(packet_path, packet.model_dump(mode="json"))
    ledger = review.dependency_ledger(packet, packet_path, "a" * 64, [], {}, "SME")
    base = build_code_index_artifact(analysis, embedding_model="text-embedding-3-large", dependency_review_ledger=ledger)
    base, _ = embed_code_index_artifact(base, client=Provider())
    base_path = tmp_path / "base_embedded.json"
    write(base_path, base.model_dump(mode="json"))
    ledger_path = tmp_path / "base_ledger.json"
    write(ledger_path, ledger.model_dump(mode="json"))
    lineage = build_lineage_artifact(fdd_generation="v9", code_artifact=base, mappings=[],
        reviewer="SME", review_packet_sha256="b" * 64, source_candidate_artifact_identity_sha256="c" * 64)
    lineage_path = tmp_path / "base_lineage.json"
    write(lineage_path, lineage.model_dump(mode="json"))
    external = tmp_path / "external"
    external.mkdir()
    (external / "pkg_customer_custom.sql").write_text(source, encoding="utf-8", newline="")
    (external / "pkg_extra.sql").write_text(source.replace("pkg_customer_custom", "pkg_extra"), encoding="utf-8", newline="")
    run = Coordinator(tmp_path, "R4")
    run.dir.mkdir(parents=True)
    fdd_directory = tmp_path / "fdd/processed"
    write(fdd_directory / "R1.retrieval_ready.json", {"document_id": "R1", "document_name": "R1.docx", "document_family": "Customer", "release_label": "R1", "units": [
        {"unit_id": "fdd-1", "unit_index": 0, "source_kind": "paragraph", "text": "Explain update_customer in pkg_customer_custom.sql", "document_id": "R1"}]})
    from app.fdd_code_lineage.evaluation import CodeCombinedEvalCase
    baseline_case = CodeCombinedEvalCase(case_id="baseline-code", mode="code", question="Explain update_customer in pkg_customer_custom.sql",
        expected_code_paths=("pkg_customer_custom.sql",), sme_reviewed=True, review_status="reviewed", rationale="Reviewed synthetic regression fixture.")
    code_cases = tmp_path / "code_cases.jsonl"
    code_cases.write_text(baseline_case.model_dump_json() + "\n", encoding="utf-8")
    combined_cases = tmp_path / "combined_cases.jsonl"
    combined_cases.write_text(baseline_case.model_copy(update={"case_id": "baseline-combined", "mode": "combined", "expected_fdd_document_ids": ("R1",)}).model_dump_json() + "\n", encoding="utf-8")
    request = SnapshotRequest(module_set="fci-custom", svn_revision="154", application_build="14.7", reviewer="SME", base_snapshot_id=base.snapshot_id)
    run.config = {"request": request.model_dump(mode="json"), "source_directory": str(external),
        "snapshot_root": str(base_dir.parent), "parse_generation": PARSER_GENERATION_DIRECTORY,
        "model": "text-embedding-3-large", "base_artifact": str(base_path), "base_analysis": str(analysis),
        "base_dependency_ledger": str(ledger_path), "base_lineage": str(lineage_path),
        "price_per_million": "0.13", "pricing_basis": "test only", "code_store": str(tmp_path / "store"),
        "fdd_generation": "v9", "fdd_stage": str(tmp_path / "fdd"), "fdd_directory": str(tmp_path / "fdd/processed"), "registry": str(tmp_path / "registry.json"),
        "code_evals": [str(code_cases)], "combined_evals": [str(combined_cases)]}
    write(run.path("config.json"), run.config)
    write(run.path("intake/snapshot_request.json"), request.model_dump(mode="json"))
    run.run.state["base_snapshot_id"] = base.snapshot_id
    run.run.save()
    def commands(script, *arguments, **kwargs):
        if script.endswith("parse_code_snapshot.py"):
            parse_code_snapshot(base_dir.parent / arguments[0], run.path("parse"))
        elif script.endswith("check_code_preindex_gate.py"):
            with pytest.raises(SystemExit) as exit_info:
                check_parse(list(map(str, arguments)))
            assert exit_info.value.code == 0
        elif script.endswith("run_code_combined_retrieval_eval.py"):
            from scripts.run_code_combined_retrieval_eval import main
            return main(list(map(str, arguments)))
        else:
            pytest.fail("Unexpected subprocess " + script)
        return 0
    monkeypatch.setattr(run, "command", commands)
    monkeypatch.setattr("app.embeddings.client.get_embedding_client", lambda **_: pytest.fail("prepare called provider"))
    run.prepare(SimpleNamespace())
    assert run.run.state["status"] == "AWAITING_EMBEDDING_APPROVAL"
    assert read(run.path("embedding_request.json"))["cached_records"] == len(base.records)
    run.prepare(SimpleNamespace())
    monkeypatch.setattr("app.embeddings.client.get_embedding_client", lambda **_: Provider())
    monkeypatch.setattr("app.fdd_code_lineage.semantic_proposals.build_proposals", lambda **_: {"documents": []})
    plan = read(run.path("embedding_request.json"))
    run.prompt = lambda _: "EMBED " + plan["request_hash"]
    run.build(SimpleNamespace(max_usd="1"))
    assert run.run.state["status"] == "AWAITING_SME_REVIEW"
    assert read(run.path("review_evidence.json"))
    resumed = Coordinator(tmp_path, "R4", prompt=lambda _: pytest.fail("repeat prompted"))
    monkeypatch.setattr("app.embeddings.client.get_embedding_client", lambda **_: pytest.fail("repeat called provider"))
    resumed.build(SimpleNamespace())
    assert resumed.run.state["status"] == "AWAITING_SME_REVIEW"
    review_path = resumed.path("review.md")
    review_path.write_text(review_path.read_text(encoding="utf-8").replace("Decision: pending", "Decision: accepted").replace(
        "Rationale: \n", "Rationale: Reviewed synthetic source and boundary expectations.\n"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("test project", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("test lock", encoding="utf-8")
    monkeypatch.setattr(resumed, "command", commands)
    def uat(final, collection, label):
        # Transport is covered separately; verify coordinator supplies FINAL inputs.
        from app.code_indexing.contract import load_code_index_artifact
        candidate = load_code_index_artifact(Path(final["code"]))
        assert candidate.dependency_review_status == "reviewed"
        output = resumed.path(label + "_mcp_uat.json")
        write(output, {"passed": True, "artifact_identity": candidate.artifact_identity_sha256, "collection": collection})
        return {"passed": True}, [output]
    monkeypatch.setattr(resumed, "local_uat", uat)
    def promotion_prepare(**kwargs):
        from app.activation.code_generation import verify_report
        from app.code_indexing.contract import load_code_index_artifact
        from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
        candidate = load_code_index_artifact(kwargs["code_artifact"])
        verify_report(kwargs["code_report"], artifact=candidate)
        verify_report(kwargs["combined_report"], artifact=candidate, lineage=load_reviewed_lineage(kwargs["lineage_path"]))
        return {"evidence_files": {}, "target_configuration": {"CODE_QDRANT_COLLECTION_NAME": kwargs["collection"]}}
    monkeypatch.setattr("app.activation.code_generation.prepare", promotion_prepare)
    resumed.finalize(SimpleNamespace())
    assert resumed.run.state["status"] == "READY_FOR_PROMOTION"
    assert not resumed.path("activation.json").exists()
    assert "final_" in resumed.run.state["final_collection"]
    from app.code_indexing.contract import load_code_index_artifact
    final_code = load_code_index_artifact(Path(resumed.run.state["final"]["code"]))
    assert all(r.embedding_status == "cached" for r in final_code.records)
    assert read(resumed.run.state["promotion_request"])["target_configuration"]["CODE_QDRANT_COLLECTION_NAME"] == resumed.run.state["final_collection"]
