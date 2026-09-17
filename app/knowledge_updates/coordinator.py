"""Resumable coordinator for compatible FDD, code, and lineage releases."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from app.code_indexing.contract import load_code_index_artifact
from app.code_updates.storage import atomic_text, digest, immutable, now, read, sha
from app.core.config import Settings
from app.fdd_code_lineage.models import validate_lineage_artifact
from app.fdd_code_lineage.reviewed_bundle import build_bundle, load_reviewed_lineage
from app.knowledge_updates.fdd_sources import plan_additive_fdd_update
from app.knowledge_updates.fdd_stage import materialize_sources, plan_fdd_embeddings
from app.knowledge_updates.models import make_release_manifest
from app.knowledge_updates.storage import Run


class Coordinator:
    """Own cross-lane state; use the established coordinator for code snapshots.

    The public workflow never allows its child code workflow to activate.  The
    parent is the only component that produces a release manifest and changes
    the compatible FDD/code/lineage selection.
    """

    def __init__(self, root: Path, run_id: str, prompt=input):
        self.root = Path(root).resolve()
        self.run = Run(self.root, run_id)
        self.directory = self.run.directory
        self._prompt = prompt
        self.config = read(self.directory / "config.json") if (self.directory / "config.json").is_file() else None

    def path(self, name: str) -> Path:
        return self.directory / name

    def ask(self, question: str) -> str:
        import time
        started = time.monotonic()
        try:
            return self._prompt(question)
        finally:
            self.run.state["operator_wait_seconds"] = self.run.state.get("operator_wait_seconds", 0.0) + time.monotonic() - started
            self.run.save()

    def value(self, args, name: str, label: str) -> str:
        answer = getattr(args, name, None) or self.ask(label + ": ").strip()
        if not answer:
            raise ValueError(f"{name.replace('_', ' ')} is required")
        return str(answer)

    def init(self, args):
        if self.config:
            self.run.check_bindings()
            print("Knowledge update already initialized; configuration retained.")
            return
        mode = getattr(args, "mode", None)
        if mode not in {"fdd", "code", "both", "review"}:
            raise ValueError("Mode is required: fdd, code, both, or review")
        settings = Settings()
        reviewer = self.value(args, "reviewer", "Reviewer name")
        price = self.value(args, "price_per_million", "Confirmed USD price per million embedding input tokens")
        pricing_basis = self.value(args, "pricing_basis", "Pricing source and date")
        from decimal import Decimal
        if not Decimal(price).is_finite() or Decimal(price) <= 0:
            raise ValueError("PricePerMillion must be a positive decimal")
        fdd_generation = getattr(args, "fdd_generation", None) or settings.fdd_generation
        fdd_source = None
        if mode in {"fdd", "both"}:
            if fdd_generation == settings.fdd_generation:
                raise ValueError("FddGeneration must be a new target for an additive FDD update")
            fdd_source = Path(self.value(args, "fdd_source_directory", "Additive FDD update directory")).resolve(strict=True)
            if not fdd_source.is_dir(): raise ValueError("FddSourceDirectory must be a directory")
        code_source = revision = build = None
        if mode in {"code", "both"}:
            code_source = Path(self.value(args, "code_source_directory", "Complete code source directory")).resolve(strict=True)
            if not code_source.is_dir(): raise ValueError("CodeSourceDirectory must be a directory")
            revision = self.value(args, "svn_revision", "Actual SVN revision")
            build = self.value(args, "application_build", "Application build")
        artifact = load_code_index_artifact(settings.code_index_artifact_path)
        lineage = load_reviewed_lineage(settings.fdd_code_lineage_artifact_path)
        registry = Path(getattr(args, "enhancement_registry", "") or self.path("empty_enhancement_registry.json"))
        if registry == self.path("empty_enhancement_registry.json"):
            immutable(registry, {"schema_version": "enhancement_fdd_registry_v1", "mappings": []})
        else:
            registry = registry.resolve(strict=True)
        replacement_manifest = getattr(args, "replacement_manifest", None)
        replacement_path = Path(replacement_manifest).resolve(strict=True) if replacement_manifest else None
        if replacement_path is not None and not replacement_path.is_file():
            raise ValueError("ReplacementManifest must be a JSON file")
        config = {
            "schema_version": "knowledge_update_config_v1", "run_id": self.run.state["run_id"], "mode": mode,
            "reviewer": reviewer, "price_per_million": str(Decimal(price)), "pricing_basis": pricing_basis,
            "base_fdd_generation": settings.fdd_generation, "base_fdd_archive": str(settings.embedded_docs_dir.resolve()),
            "base_fdd_directory": str(settings.fdd_retrieval_artifact_dir.resolve()),
            "base_code_artifact": str(settings.code_index_artifact_path.resolve()),
            "base_code_analysis": str(settings.code_analysis_directory.resolve()),
            "base_code_collection": settings.code_qdrant_collection_name, "base_code_snapshot_id": artifact.snapshot_id,
            "base_lineage": str(settings.fdd_code_lineage_artifact_path.resolve()), "base_lineage_identity": lineage.artifact_identity_sha256,
            "fdd_generation": fdd_generation, "fdd_source_directory": str(fdd_source) if fdd_source else None,
            "code_source_directory": str(code_source) if code_source else None, "svn_revision": revision,
            "application_build": build, "code_child_run_id": f"{self.run.state['run_id']}-code" if code_source else None,
            "enhancement_registry": str(registry), "replacement_manifest": str(replacement_path) if replacement_path else None,
        }
        immutable(self.path("config.json"), config); self.config = config
        self.run.bind([self.path("config.json"), Path(config["base_code_artifact"]), Path(config["base_lineage"]), registry,
                       *([replacement_path] if replacement_path else [])])
        self.run.bind_tree(Path(config["base_fdd_directory"]), "*.retrieval_ready.json")
        self.run.transition("INITIALIZED", "prepare")
        print(json.dumps({"mode": mode, "base_fdd_generation": config["base_fdd_generation"],
                          "target_fdd_generation": fdd_generation, "base_code_snapshot_id": artifact.snapshot_id}, indent=2))

    def prepare(self, args):
        self.require("INITIALIZED", "AWAITING_EMBEDDING_APPROVAL")
        if self.config["mode"] in {"fdd", "both"}:
            def operation():
                replacements = ()
                replacement_manifest = self.config.get("replacement_manifest")
                if replacement_manifest:
                    payload = read(Path(replacement_manifest))
                    if payload.get("schema_version") != "knowledge_fdd_replacement_manifest_v1" or not isinstance(payload.get("replacements"), list):
                        raise ValueError("ReplacementManifest must use knowledge_fdd_replacement_manifest_v1 with a replacements list")
                    replacements = tuple(str(item) for item in payload["replacements"])
                plan = plan_additive_fdd_update(base_generation=self.config["base_fdd_generation"], target_generation=self.config["fdd_generation"],
                    baseline_directory=Path(self.config["base_fdd_archive"]), update_directory=Path(self.config["fdd_source_directory"]),
                    withdrawn=([getattr(args, "withdrawn_source")] if getattr(args, "withdrawn_source", None) else ()), replacements=replacements)
                plan_path = self.path("fdd_plan.json"); immutable(plan_path, plan.model_dump(mode="json"))
                intake = self.path("intake/fdd/source")
                if not intake.exists():
                    materialize_sources(plan=plan, baseline_directory=Path(self.config["base_fdd_archive"]), update_directory=Path(self.config["fdd_source_directory"]), destination=intake)
                self.run.bind_tree(intake, "*.docx")
                request = plan_fdd_embeddings(source_directory=intake, cache_directories=[self.root / "data/cache/embeddings"],
                    embedding_model=Settings().openai_embedding_model, artifact_version=Settings().artifact_version,
                    usd_per_million_tokens=self.config["price_per_million"], pricing_basis=self.config["pricing_basis"])
                request_path = self.path("fdd_embedding_request.json"); immutable(request_path, request)
                return {"changes": plan.changes.model_dump(mode="json"), "fdd_embedding_inputs": request["unique_missing_inputs"]}, [plan_path, request_path]
            self.run.step("fdd_prepare", operation)
        # A combined update stages the target FDD generation before its code
        # child is initialized.  The code child binds that verified target as
        # evidence; pointing it at the live baseline would miss new-FDD to
        # unchanged-code candidates.
        if self.config["mode"] == "code": self.code_child("prepare", args)
        self.run.transition("AWAITING_EMBEDDING_APPROVAL", "build")

    def build(self, args):
        self.require("AWAITING_EMBEDDING_APPROVAL", "AWAITING_SME_REVIEW")
        if self.config["mode"] in {"fdd", "both"}:
            request = read(self.path("fdd_embedding_request.json"))
            if request["unique_missing_inputs"]:
                exact = "EMBED " + request["request_hash"]
                approval = getattr(args, "embedding_approval", None) or self.ask(
                    f"FDD lane has {request['unique_missing_inputs']} uncached inputs; upper bound USD {request['cost_upper_bound_usd']}. Type {exact}: ")
                if approval != exact: raise PermissionError("Exact FDD embedding approval is required")
                maximum = getattr(args, "max_usd", None) or self.ask("Maximum USD authorized for this run: ")
                from decimal import Decimal
                if Decimal(str(maximum)) < Decimal(request["cost_upper_bound_usd"]): raise PermissionError("Embedding ceiling is below frozen FDD request")
                immutable(self.path("approvals/fdd_embedding.json"), {"request_hash": request["request_hash"], "max_usd": str(maximum), "approved_by": self.config["reviewer"], "at": now()})
            stage = self.path("fdd_stage")
            if not stage.exists():
                command = [sys.executable, str(self.root / "scripts/stage_archived_fdd_rebuild.py"), "--source-directory", str(self.path("intake/fdd/source")),
                    "--stage-directory", str(stage), "--collection-name", self.config["fdd_generation"], "--index-generation", self.config["fdd_generation"]]
                if subprocess.run(command, cwd=self.root).returncode: raise RuntimeError("FDD staged rebuild failed; inspect its stage manifest then resume build")
            manifest = stage / "stage_manifest.json"
            if read(manifest).get("status") != "verified": raise RuntimeError("FDD stage did not reach verified state")
            self.run.step("fdd_stage", lambda: ({"stage": str(stage)}, [manifest]))
            self.make_fdd_review()
        if self.config["mode"] == "both": self.code_child("prepare", args)
        if self.config["mode"] in {"code", "both"}: self.code_child("build", args)
        self.run.transition("AWAITING_SME_REVIEW", "finalize")

    def make_fdd_review(self):
        from app.code_updates import review
        from app.fdd_code_lineage.semantic_proposals import build_proposals
        artifact = load_code_index_artifact(Path(self.config["base_code_artifact"]))
        proposals_path = self.path("fdd_proposals.json")
        if not proposals_path.exists():
            proposal = build_proposals(fdd_stage=self.path("fdd_stage"), snapshot_directory=Settings().code_snapshots_dir / artifact.snapshot_id,
                analysis_directory=Path(self.config["base_code_analysis"]), code_artifact_path=Path(self.config["base_code_artifact"]),
                registry_path=Path(self.config["enhancement_registry"]), allow_provisional=True, progress=lambda text: print(text, flush=True))
            immutable(proposals_path, proposal)
        plan, proposals = read(self.path("fdd_plan.json")), read(proposals_path)
        changed_docs = {Path(path).stem for path in plan["changes"]["added"] + plan["changes"]["modified"]}
        inactive_docs = {Path(path).stem for path in plan["changes"]["superseded"] + plan["changes"]["withdrawn"]}
        base = load_reviewed_lineage(Path(self.config["base_lineage"]))
        items = []
        for mapping in base.mappings:
            carry = mapping.fdd_document_id not in changed_docs | inactive_docs
            evidence = {"fdd_document_id": mapping.fdd_document_id, "fdd_release_label": mapping.fdd_release_label,
                "targets": [target.model_dump(mode="json") for target in mapping.targets], "prior_mapping": mapping.model_dump(mode="json"),
                "base_lineage_identity": base.artifact_identity_sha256}
            if carry: evidence["carry_decision"] = {"verdict": "accepted", "rationale": mapping.rationale, "correction": {},
                "prior_reviewer": mapping.reviewer, "prior_mapping_id": mapping.mapping_id, "prior_lineage_identity": base.artifact_identity_sha256}
            items.append(review.item("inherited_lineage", evidence, "Does this prior FDD/code relationship remain valid?",
                                     "validated_carry_forward" if carry else "prior_human_review"))
        existing = {(m.fdd_document_id, t.path, t.qualified_name) for m in base.mappings for t in m.targets}
        for document in proposals["documents"]:
            if document["document_id"] not in changed_docs: continue
            release = re.search(r"(?:^|_)R(\d+)(?:_|$)", document["document_id"])
            for candidate in document["candidates"]:
                target = candidate["target"]
                if (document["document_id"], target["path"], target["qualified_name"]) in existing: continue
                evidence = {"fdd_document_id": document["document_id"], "fdd_release_label": "R" + release.group(1) if release else "unlabelled",
                    "targets": [{**target, "rationale": "Stored-vector/source-context candidate; SME confirmation is required."}], "proposal": candidate,
                    "evaluation_question": f"What FDD requirement is implemented by {target['qualified_name']} in {target['path']}?"}
                items.append(review.item("lineage", evidence, "Does this routine implement/support this FDD passage? Defer if uncertain."))
        immutable(self.path("fdd_review_evidence.json"), items); review.render(items, self.path("review.md"))

    def finalize(self, args):
        self.require("AWAITING_SME_REVIEW", "READY_FOR_PROMOTION")
        if self.config["mode"] == "review":
            # A review-only run has no source, metadata, collection, or
            # configuration change. It records a completed no-op rather than
            # manufacturing a fresh activation request for an unchanged state.
            immutable(self.path("review_only_result.json"), {"status": "complete_noop", "at": now(),
                "reason": "No source or reviewed-decision amendment was supplied"})
            self.run.transition("COMPLETE", "status")
            return
        fdd = self.finalize_fdd_review() if self.config["mode"] in {"fdd", "both"} else None
        if self.config["mode"] in {"code", "both"}: self.code_child("finalize", args)
        final = self.combination(fdd)
        self.final_gates(final)
        self.create_promotion(final)
        self.run.transition("READY_FOR_PROMOTION", "activate")

    def finalize_fdd_review(self):
        from app.code_updates import review
        artifact = load_code_index_artifact(Path(self.config["base_code_artifact"]))
        items = read(self.path("fdd_review_evidence.json")); _, selected = review.collect_review_decisions(items, self.path("review.md"))
        review_hash = review.combined_review_identity(self.path("review.md"))
        overrides, _ = review.normalize_lineage_targets(items, selected, artifact)
        candidate, reviewed, deferred = review.finalize_lineage(items, selected, artifact, self.config["fdd_generation"], review_hash, self.config["reviewer"], target_overrides=overrides)
        known = {path.name.removesuffix(".retrieval_ready.json") for path in self.path("fdd_stage/processed").glob("*.retrieval_ready.json")}
        validate_lineage_artifact(reviewed, fdd_document_ids=known, code_artifact=artifact, analysis_directory=Path(self.config["base_code_analysis"]))
        directory = self.path("final/fdd_" + review_hash[:16]); directory.mkdir(parents=True, exist_ok=True)
        candidate_path, lineage_path, deferred_path = directory / "candidate_lineage.json", directory / "reviewed_lineage.json", directory / "deferred_lineage.json"
        immutable(candidate_path, candidate.model_dump(mode="json")); immutable(lineage_path, reviewed.model_dump(mode="json")); immutable(deferred_path, deferred)
        immutable(directory / "decisions.json", {"review_hash": review_hash, "reviewer": self.config["reviewer"], "decisions": selected})
        result = {"candidate": str(candidate_path), "lineage": str(lineage_path), "deferred": str(deferred_path), "review_hash": review_hash}
        self.run.step("fdd_finalize_" + review_hash[:16], lambda: (result, [p for p in directory.rglob("*") if p.is_file()]))
        return result

    def combination(self, fdd):
        mode = self.config["mode"]
        if mode in {"code", "both"}:
            child_dir = self.root / "data/code_updates" / self.config["code_child_run_id"]
            child = read(child_dir / "state.json")
            if child.get("status") != "READY_FOR_PROMOTION" or not child.get("final"): raise RuntimeError("Code child finalization is incomplete")
            code, analysis, collection, code_lineage = child["final"]["code"], child["analysis"], child["final_collection"], child["final"]["lineage"]
        else:
            code, analysis, collection, code_lineage = self.config["base_code_artifact"], self.config["base_code_analysis"], self.config["base_code_collection"], None
        if mode == "both":
            from app.fdd_code_lineage.reviewed_bundle import write_bundle
            output = self.path("final/lineage_bundle/reviewed_lineage_bundle.json")
            if not output.exists():
                code_final = Path(code_lineage).parent
                write_bundle(build_bundle([(Path(fdd["lineage"]), Path(fdd["candidate"]), self.path("review.md")),
                    (Path(code_lineage), code_final / "candidate_lineage.json", self.root / "data/code_updates" / self.config["code_child_run_id"] / "review.md")]), output)
            lineage, deferred = str(output), fdd["deferred"]
        elif mode == "fdd": lineage, deferred = fdd["lineage"], fdd["deferred"]
        else: lineage, deferred = code_lineage, str(Path(code_lineage).parent / "deferred_lineage.json")
        staged = mode in {"fdd", "both"}
        return {"code": code, "analysis": analysis, "code_collection": collection, "lineage": lineage, "deferred": deferred,
            "fdd_generation": self.config["fdd_generation"], "fdd_collection": self.config["fdd_generation"] if staged else Settings().qdrant_collection_name,
            "fdd_directory": str(self.path("fdd_stage/processed")) if staged else self.config["base_fdd_directory"],
            "fdd_stage_manifest": str(self.path("fdd_stage/stage_manifest.json")) if staged else str(self.root / "data/staging" / self.config["base_fdd_generation"] / "stage_manifest.json")}

    def final_gates(self, final):
        reports = []
        fdd_eval = self.root / "data/evaluations/fdd_grounded_eval_v2_reviewed.jsonl"
        if fdd_eval.is_file():
            output = self.path("final/fdd_retrieval_evaluation.json")
            if not output.exists():
                command = [sys.executable, str(self.root / "scripts/run_fdd_retrieval_gate.py"), "--eval-file", str(fdd_eval), "--collection-name", final["fdd_collection"], "--lexical-artifact-directory", final["fdd_directory"], "--output-file", str(output)]
                # Ingestion approval never authorizes paid query embeddings.
                import os
                environment = dict(os.environ, RETRIEVAL_MODE="lexical")
                if subprocess.run(command, cwd=self.root, env=environment).returncode: raise RuntimeError(f"Final FDD retrieval gate failed: {output}")
            reports.append(output)
        combined_eval = self.root / "data/evaluations/combined_grounded_eval_v2_reviewed.jsonl"
        if combined_eval.is_file():
            output = self.path("final/combined_retrieval_evaluation.json")
            if not output.exists():
                command = [sys.executable, str(self.root / "scripts/run_code_combined_retrieval_eval.py"), "--eval-file", str(combined_eval),
                    "--code-artifact", final["code"], "--analysis-directory", final["analysis"], "--fdd-generation", final["fdd_generation"],
                    "--fdd-directory", final["fdd_directory"], "--lineage-artifact", final["lineage"], "--code-mode", "lexical", "--output-file", str(output)]
                if subprocess.run(command, cwd=self.root).returncode: raise RuntimeError(f"Final combined regression gate failed: {output}")
            reports.append(output)
        if not reports: raise RuntimeError("No reviewed final evaluation manifest is available; promotion is blocked")
        index = self.path("final/evaluation_index.json"); immutable(index, {"reports": [str(p) for p in reports], "sha256": {str(p): sha(p) for p in reports}})
        self.run.step("final_gates", lambda: ({"reports": [str(p) for p in reports]}, [*reports, index]))
        self.run_local_mcp_uat(final, combined_eval)

    def run_local_mcp_uat(self, final, combined_eval: Path):
        """Exercise the exact staged combination without touching live settings."""
        output = self.path("final/mcp_uat.json")
        if output.exists():
            if not read(output).get("passed"): raise ValueError("Recorded MCP UAT did not pass")
            return
        from app.code_updates.runtime import staged_uat
        child_state = {"run_id": self.run.state["run_id"], "analysis": final["analysis"],
                       "steps": {"snapshot": {"result": {"diff": {"added": [], "modified": [], "deleted": []}}}}}
        if self.config["code_child_run_id"]:
            child_state = read(self.root / "data/code_updates" / self.config["code_child_run_id"] / "state.json")
        config = {"code_store": str(Settings().code_qdrant_local_path), "fdd_generation": final["fdd_generation"],
                  "fdd_directory": final["fdd_directory"], "combined_evals": [str(combined_eval)] if combined_eval.is_file() else []}
        try:
            result = staged_uat(self.root, config, child_state,
                                {"code": final["code"], "lineage": final["lineage"]}, final["code_collection"])
        except RuntimeError as exc:
            raise RuntimeError(f"Bounded local MCP UAT failed for the final staged combination: {exc}") from exc
        if not result.get("passed"): raise RuntimeError("Bounded local MCP UAT did not pass")
        immutable(output, result)

    def create_promotion(self, final):
        lineage, artifact = load_reviewed_lineage(Path(final["lineage"])), load_code_index_artifact(Path(final["code"]))
        runtime_files = __import__("app.activation.code_generation", fromlist=["runtime_files"]).runtime_files(self.root)
        manifest = make_release_manifest(run_id=self.run.state["run_id"], mode=self.config["mode"], fdd_generation=final["fdd_generation"],
            fdd_collection=final["fdd_collection"], fdd_processed_directory=final["fdd_directory"], fdd_stage_manifest_sha256=sha(final["fdd_stage_manifest"]),
            fdd_stage_directory=str(Path(final["fdd_stage_manifest"]).parent) if self.config["mode"] in {"fdd", "both"} else None,
            code_snapshot_id=artifact.snapshot_id, code_collection=final["code_collection"], code_artifact_path=final["code"],
            code_artifact_identity_sha256=artifact.artifact_identity_sha256, code_analysis_directory=final["analysis"], lineage_path=final["lineage"],
            lineage_identity_sha256=lineage.artifact_identity_sha256, deferred_lineage_path=final["deferred"],
            review_identity_sha256=digest({"fdd": sha(self.path("review.md")) if self.path("review.md").exists() else None,
                "code": sha(self.root / "data/code_updates" / self.config["code_child_run_id"] / "review.md") if self.config["code_child_run_id"] else None}),
            evaluation_report_sha256=sha(self.path("final/evaluation_index.json")), runtime_files_sha256=digest(runtime_files))
        manifest_path = self.path("release_manifest.json"); immutable(manifest_path, manifest.model_dump(mode="json"))
        from app.activation.knowledge_release import prepare
        request = prepare(root=self.root, manifest_path=manifest_path, requested_by=self.config["reviewer"])
        request_path = self.path("promotion_request.json"); immutable(request_path, request)
        self.run.state.update(final=final, release_manifest=str(manifest_path), promotion_request=str(request_path), promotion_hash=request["request_identity_sha256"]); self.run.save()

    def activate(self, args):
        if not self.run.state.get("promotion_request"): raise ValueError("Finalize successfully before activation")
        if not getattr(args, "services_stopped", False): raise ValueError("Stop Desktop MCP/FastAPI then repeat with -ServicesStopped")
        from app.activation.code_modes import ActivationApproval, build_activation_approval
        from app.activation.knowledge_release import switch
        request, approval_path = read(self.run.state["promotion_request"]), self.path("promotion_approval.json")
        if not approval_path.exists():
            exact = "ACTIVATE " + request["request_identity_sha256"]
            if self.ask(f"Approve compatible FDD/code/lineage promotion by typing {exact}: ").strip() != exact: raise PermissionError("Exact promotion approval is required")
            approval = build_activation_approval(request=SimpleNamespace(request_identity_sha256=request["request_identity_sha256"]), approved_by=self.config["reviewer"], paid_smoke_authorized=False, internal_evidence_disclosure_authorized=False)
            immutable(approval_path, approval.model_dump(mode="json"))
        from app.code_updates.runtime import stopped_mcp_guard
        with stopped_mcp_guard(): result = switch(root=self.root, request=request, approval=ActivationApproval.model_validate(read(approval_path)), action="activate", apply=True)
        immutable(self.path("activation.json"), {**result, "at": now()})
        from app.knowledge_updates.runtime import create_restart_challenge
        create_restart_challenge(self.run, read(self.run.state["release_manifest"]))
        self.run.transition("AWAITING_RESTART", "verify")

    def verify(self, args):
        receipt = getattr(args, "runtime_receipt", None)
        if not receipt:
            candidates = list(self.path("runtime_receipts").glob("*.json"))
            if len(candidates) != 1: raise ValueError("Restart MCP then run verify again; no unique knowledge runtime receipt is available")
            receipt = candidates[0]
        from app.knowledge_updates.runtime import verify_restart_receipt
        verify_restart_receipt(self.run, Path(receipt).resolve(strict=True))
        immutable(self.path("runtime_receipt.json"), read(Path(receipt).resolve(strict=True))); self.run.transition("COMPLETE", "status")

    def rollback(self, args):
        if not getattr(args, "services_stopped", False): raise ValueError("Stop Desktop MCP/FastAPI before rollback")
        from app.activation.code_modes import ActivationApproval
        from app.activation.knowledge_release import switch
        from app.code_updates.runtime import stopped_mcp_guard
        with stopped_mcp_guard(): result = switch(root=self.root, request=read(self.run.state["promotion_request"]), approval=ActivationApproval.model_validate(read(self.path("promotion_approval.json"))), action="rollback", apply=True)
        immutable(self.path("rollback.json"), {**result, "at": now()}); self.run.transition("AWAITING_RESTART", "verify")

    def amend_review(self, args): raise ValueError("Use a new knowledge run for a changed review decision; immutable evidence is retained")

    def code_child(self, action, args):
        run_id, directory = self.config["code_child_run_id"], self.root / "data/code_updates" / self.config["code_child_run_id"]
        if action == "prepare" and not (directory / "config.json").exists():
            init = [sys.executable, str(self.root / "scripts/run_code_update.py"), "--action", "init", "--run-id", run_id, "--source-directory", self.config["code_source_directory"], "--svn-revision", self.config["svn_revision"], "--application-build", self.config["application_build"], "--reviewer", self.config["reviewer"], "--price-per-million", self.config["price_per_million"], "--pricing-basis", self.config["pricing_basis"]]
            if self.config["mode"] == "both": init += ["--fdd-generation", self.config["fdd_generation"], "--fdd-directory", str(self.path("fdd_stage/processed")), "--fdd-stage", str(self.path("fdd_stage")), "--coordinated-release"]
            if subprocess.run(init, cwd=self.root).returncode: raise RuntimeError("Code child initialization failed")
        command = [sys.executable, str(self.root / "scripts/run_code_update.py"), "--action", action, "--run-id", run_id]
        if getattr(args, "max_usd", None): command += ["--max-usd", str(args.max_usd)]
        if subprocess.run(command, cwd=self.root).returncode: raise RuntimeError(f"Code child {action} failed; inspect data/code_updates/{run_id}")

    def require(self, *statuses):
        if self.run.state.get("status") not in statuses: raise ValueError(f"Action is not valid from {self.run.state.get('status')}; expected {', '.join(statuses)}")

    def status(self, args):
        print(json.dumps(self.run.state, indent=2))
        for candidate in (self.path("review.md"), self.root / "data/code_updates" / str(self.config.get("code_child_run_id")) / "review.md" if self.config else None):
            if candidate and candidate.is_file(): print(f"Review packet: {candidate}")
        print(f"Next: .\\scripts\\run_knowledge_update.ps1 -Action {self.run.state.get('next_action', 'init')} -RunId {self.run.state['run_id']}")

    def write_summary(self):
        text = f"# Knowledge update {self.run.state['run_id']}\n\nStatus: **{self.run.state.get('status')}**\n\nNext: `.\\scripts\\run_knowledge_update.ps1 -Action {self.run.state.get('next_action', 'init')} -RunId {self.run.state['run_id']}`\n"
        if self.config: text += f"\n- Mode: {self.config['mode']}\n- FDD: {self.config['base_fdd_generation']} -> {self.config['fdd_generation']}\n- Base code: {self.config['base_code_snapshot_id']}\n"
        atomic_text(self.path("summary.md"), text)
        import html
        atomic_text(self.path("summary.html"), "<!doctype html><meta charset=\"utf-8\"><pre>" + html.escape(text) + "</pre>")

    def execute(self, action, args): return getattr(self, action.replace("-", "_"))(args)
