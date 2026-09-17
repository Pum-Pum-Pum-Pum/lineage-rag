"""Recurring update orchestration. Every paid or serving transition is explicit."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from app.code_updates.storage import Run, read, write, immutable, atomic_text, digest, sha, now
from app.code_updates.embedding import plan_embeddings, checkpointed_embed, positive_decimal, approved_client
from app.code_updates import review
from app.code_indexing.contract import build_code_index_artifact, load_code_index_artifact
from app.code_ingestion.snapshot_builder import load_snapshot_manifest, build_code_snapshot
from app.code_ingestion.snapshot_models import SnapshotRequest
from app.core.config import Settings


class EmbeddingApprovalRequired(Exception):
    """A reviewed metadata change introduced explicitly unapproved input text."""


def dump_model(path, model):
    immutable(path, model.model_dump(mode="json"))
    return str(path)


def active_baseline(root, settings):
    from app.activation.code_generation import KEYS, digest as promotion_digest
    target = {
        "CODE_MODES_ENABLED": str(settings.code_modes_enabled).lower(),
        "CODE_INDEX_ARTIFACT_PATH": str(Path(settings.code_index_artifact_path).resolve()),
        "CODE_ANALYSIS_DIRECTORY": str(Path(settings.code_analysis_directory).resolve()),
        "CODE_QDRANT_COLLECTION_NAME": settings.code_qdrant_collection_name,
        "FDD_CODE_LINEAGE_ARTIFACT_PATH": str(Path(settings.fdd_code_lineage_artifact_path).resolve()),
    }
    receipts = []
    directory = root / "data/exports/activation"
    for p in directory.glob("*.result.json"):
        value = read(p)
        if value.get("applied") and value.get("action") == "activate":
            receipts.append(value.get("request_identity_sha256"))
    matches = []
    for p in directory.glob("*-request.json"):
        value = read(p)
        if value.get("schema_version") != "code_generation_promotion_request_v1":
            continue
        selected = value.get("target_configuration", {})
        normalized = {k: str(Path(v).resolve()) if k.endswith(("PATH", "DIRECTORY")) else v for k, v in selected.items()}
        if normalized != target or value.get("request_identity_sha256") not in receipts:
            continue
        if promotion_digest({k: v for k, v in value.items() if k != "request_identity_sha256"}) != value["request_identity_sha256"]:
            raise ValueError("Active promotion request is corrupt")
        matches.append((p, value))
    if len(matches) != 1:
        raise ValueError(f"Expected one applied promotion matching active settings; found {len(matches)}. Reconcile baseline explicitly.")
    return matches[0]


def finalized_review_identity(directory: Path) -> str:
    """Return the review identity that must remain unchanged through promotion.

    A review amendment is a separately reviewed packet.  It is part of the
    finalization identity whenever its evidence exists, just as it is during
    finalization itself.  Checking only ``review.md`` here would reject a
    valid amended finalization (or, worse, fail to notice a changed amendment).
    """
    primary = directory / "review.md"
    amendment_evidence = directory / "review_amendment_evidence.json"
    amendment = directory / "review_amendment.md"
    if amendment_evidence.exists() != amendment.exists():
        raise ValueError("Review amendment evidence and Markdown packet must either both exist or both be absent")
    return review.combined_review_identity(primary, amendment if amendment_evidence.exists() else None)


class Coordinator:
    def __init__(self, root, run_id, prompt=input):
        self.run = Run(root, run_id)
        self.root, self.dir = self.run.root, self.run.directory
        def timed_prompt(question):
            started = time.monotonic()
            try:
                return prompt(question)
            finally:
                self.run.state["prompt_wait_seconds"] = self.run.state.get("prompt_wait_seconds", 0) + time.monotonic() - started
                self.run.state["operator_prompts"] = self.run.state.get("operator_prompts", 0) + 1
                self.run.save()
        self.prompt = timed_prompt
        self.config = read(self.dir / "config.json") if (self.dir / "config.json").exists() else None

    def path(self, name):
        return self.dir / name

    def command(self, script, *arguments, name=None, tolerate_failure=False):
        name = name or Path(script).stem
        log = self.path(f"logs/{name}.txt")
        log.parent.mkdir(parents=True, exist_ok=True)
        self.run.event("command", script=script, log=str(log))
        with log.open("a", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, str(self.root / script), *map(str, arguments)],
                                    cwd=self.root, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode and not tolerate_failure:
            raise RuntimeError(f"{script} failed ({result.returncode}). Read {log}; resume the same action after correction.")
        return result.returncode

    def init(self, args):
        if self.config:
            self.run.check_bindings()
            if self.run.state["status"] == "NEW":
                self.complete_init()
            print("Run already initialized; configuration retained.")
            return
        settings = Settings()
        request_path, baseline = active_baseline(self.root, settings)
        artifact = load_code_index_artifact(settings.code_index_artifact_path)
        snapshot = load_snapshot_manifest(settings.code_snapshots_dir / artifact.snapshot_id)
        def value(name, question):
            return getattr(args, name, None) or self.prompt(question + ": ").strip()
        source = Path(value("source_directory", "Complete source directory")).resolve(strict=True)
        if not source.is_dir():
            raise ValueError("SourceDirectory must be a directory")
        revision = value("svn_revision", "Actual SVN revision (not the R4 release label)")
        reviewer = value("reviewer", "Reviewer name")
        build = value("application_build", "Application build")
        price = str(positive_decimal(value("price_per_million", "Confirmed USD price per million embedding input tokens")))
        basis = value("pricing_basis", "Pricing source and date")
        if not basis:
            raise ValueError("Pricing basis is required")
        snapshot_request = SnapshotRequest(module_set=snapshot.request.module_set, svn_revision=revision,
            application_build=build, reviewer=reviewer, base_snapshot_id=artifact.snapshot_id)
        manifests = {"code": set(), "combined": set()}
        boundary_manifests = set()
        ledger_path = None
        for relative, expected in baseline["evidence_files"].items():
            p = self.root / relative
            if not p.is_file():
                raise ValueError(f"Active promotion evidence missing: {p}")
            if sha(p) != expected:
                raise ValueError(f"Active promotion evidence changed: {p}")
            if p.suffix != ".json":
                continue
            payload = read(p)
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version") == "code_documentation_boundary_eval_v1":
                meta = payload["metadata"]
                boundary_file = (self.root / meta["eval_file"]).resolve()
                if not payload["summary"]["passed"] or sha(boundary_file) != meta["eval_file_sha256"]:
                    raise ValueError("Active documentation-boundary evidence changed or failed")
                boundary_manifests.add(str(boundary_file))
                # Boundary reports have their own passed contract; they are
                # not code/combined retrieval reports with release_gate_eligible.
                continue
            if payload.get("ledger_identity_sha256") == artifact.dependency_review_ledger_sha256:
                ledger_path = p
            if "metadata" in payload and "summary" in payload and "cases" in payload:
                if sha(p) != expected or not payload["summary"].get("release_gate_eligible"):
                    raise ValueError("Active regression report is changed or failed")
                modes = {c["mode"] for c in payload["cases"]}
                if len(modes) == 1 and next(iter(modes)) in manifests:
                    for file, file_hash in payload["metadata"]["eval_file_sha256"].items():
                        eval_path = (self.root / file).resolve()
                        if sha(eval_path) != file_hash:
                            raise ValueError("Active regression manifest changed")
                        manifests[next(iter(modes))].add(str(eval_path))
        if not all(manifests.values()) or ledger_path is None:
            raise ValueError("Active promotion must identify code/combined regressions and dependency ledger")
        registry = getattr(args, "enhancement_registry", None)
        if registry:
            registry = str(Path(registry).resolve(strict=True))
        else:
            # Registry is a discovery aid, not a prerequisite for blind discovery.
            registry = str(self.path("empty_registry.json"))
            immutable(Path(registry), {"schema_version": "enhancement_fdd_registry_v1", "mappings": []})
        # The coordinated knowledge workflow may stage a new FDD generation
        # before creating a complete code snapshot.  These explicit arguments
        # are a compatibility adapter, never an inferred directory selection.
        target_fdd_generation = getattr(args, "fdd_generation", None) or settings.fdd_generation
        target_fdd_directory = getattr(args, "fdd_directory", None) or str(settings.fdd_retrieval_artifact_dir.resolve())
        target_fdd_stage = getattr(args, "fdd_stage", None) or str((settings.data_dir / "staging" / settings.fdd_generation).resolve())
        target_fdd_directory = str(Path(target_fdd_directory).resolve(strict=True))
        target_fdd_stage = str(Path(target_fdd_stage).resolve(strict=True))
        if not (Path(target_fdd_stage) / "stage_manifest.json").is_file():
            raise ValueError("Selected FDD stage lacks a verified stage manifest")
        if read(Path(target_fdd_stage) / "stage_manifest.json").get("status") != "verified":
            raise ValueError("Selected FDD stage is not verified")
        self.config = dict(schema_version="code_update_config_v1", run_id=self.run.state["run_id"],
            source_directory=str(source), request=snapshot_request.model_dump(mode="json"),
            base_artifact=str(Path(settings.code_index_artifact_path).resolve()),
            base_analysis=str(Path(settings.code_analysis_directory).resolve()),
            base_lineage=str(Path(settings.fdd_code_lineage_artifact_path).resolve()),
            base_dependency_ledger=str(ledger_path), base_promotion=str(request_path),
            fdd_generation=target_fdd_generation, fdd_directory=target_fdd_directory,
            fdd_stage=target_fdd_stage,
            code_store=str(settings.code_qdrant_local_path.resolve()),
            snapshot_root=str(settings.code_snapshots_dir.resolve()),
            parse_generation=artifact.parse_generation, model=artifact.embedding_model,
            price_per_million=price, pricing_basis=basis, registry=registry,
            code_evals=sorted(manifests["code"]), combined_evals=sorted(manifests["combined"]), boundary_evals=sorted(boundary_manifests),
            coordinated_release=bool(getattr(args, "coordinated_release", False)))
        if artifact.embedding_model != "text-embedding-3-large":
            raise ValueError("This workflow version supports text-embedding-3-large only")
        immutable(self.path("config.json"), self.config)
        self.complete_init()

    def complete_init(self):
        """Recover a crash between resolved configuration and initial checkpoint."""
        request = SnapshotRequest.model_validate(self.config["request"])
        immutable(self.path("intake/snapshot_request.json"), request.model_dump(mode="json"))
        bindings = [self.path("config.json"), self.path("intake/snapshot_request.json"),
                    Path(self.config["base_promotion"]), Path(self.config["base_dependency_ledger"]), Path(self.config["registry"]),
                    Path(self.config["base_artifact"]), Path(self.config["base_lineage"]),
                    *map(Path, self.config["code_evals"] + self.config["combined_evals"] + self.config["boundary_evals"])]
        bindings += list(Path(self.config["fdd_directory"]).glob("*.retrieval_ready.json"))
        self.run.bind(bindings)
        self.run.bind_tree(self.config["fdd_directory"], "*.retrieval_ready.json")
        self.run.bind_tree(self.config["base_analysis"], "*.json")
        self.run.bind([Path(self.config["fdd_stage"]) / "stage_manifest.json"])
        self.run.bind_tree(Path(self.config["fdd_stage"]) / "cache/embeddings", "*.json")
        self.run.state.update(base_snapshot_id=request.base_snapshot_id, revision=request.svn_revision)
        self.run.transition("INITIALIZED", "prepare")
        print(json.dumps({"source_directory": self.config["source_directory"], "actual_svn_revision": request.svn_revision,
            "base_snapshot_id": request.base_snapshot_id,
            "fdd_generation": self.config["fdd_generation"]}, indent=2))

    def prepare(self, args):
        if "embed" in self.run.state["steps"]:
            print("Preparation already consumed by build; completed state retained.")
            return
        from app.code_ingestion.source_import import stage_external_code_source, _tree_sha256
        from app.code_ingestion.snapshot_builder import _source_hashes
        from app.code_ingestion.intake_validation import validate_code_intake
        from app.code_ingestion.dependency_review import build_dependency_review_packet
        intake = self.path("intake")
        def imported():
            receipt_path = intake / "source_import_receipt.json"
            if receipt_path.exists():
                receipt = read(receipt_path)
                if _tree_sha256(_source_hashes(intake / "source")) != receipt["selected_tree_sha256"]:
                    raise ValueError("Imported source differs from receipt")
            else:
                stage_external_code_source(Path(self.config["source_directory"]), intake)
            return {}, [receipt_path, *sorted((intake / "source").rglob("*.*"))]
        self.run.step("import", imported)
        self.run.bind_tree(intake / "source")
        def snapshot_stage():
            from app.code_ingestion.snapshot_builder import _snapshot_content_hash
            request = SnapshotRequest.model_validate(self.config["request"])
            validation = validate_code_intake(intake / "source")
            fingerprint = _snapshot_content_hash(request, validation.files, ingestion_policy_sha256=validation.ingestion_policy_sha256)
            snapshot_id = f"{request.module_set}-r{request.svn_revision}-{fingerprint[:12]}"
            target = Path(self.config["snapshot_root"]) / snapshot_id
            manifest = load_snapshot_manifest(target) if target.exists() else build_code_snapshot(intake, Path(self.config["snapshot_root"]))
            return {"snapshot_id": manifest.snapshot_id, "diff": manifest.diff.model_dump(mode="json")}, [target / "snapshot_manifest.json"]
        result = self.run.step("snapshot", snapshot_stage)
        self.run.state["snapshot_id"] = result["snapshot_id"]
        self.run.save()
        delta = result["diff"]
        print(json.dumps(delta, indent=2))
        if delta["deleted"]:
            confirmation = self.path("deletion_approval.json")
            if not confirmation.exists():
                expected = f"DELETE {digest(delta['deleted'])}"
                if self.prompt(f"Confirm intended removals only by typing {expected}: ").strip() != expected:
                    raise ValueError("Source deletions need explicit reconciliation")
                immutable(confirmation, {"paths": delta["deleted"], "reviewer": self.config["request"]["reviewer"]})
            if read(confirmation)["paths"] != delta["deleted"]:
                raise ValueError("Deletion approval does not match snapshot")
        if not any(delta[key] for key in ("added", "modified", "deleted")):
            self.run.transition("NO_SOURCE_CHANGES", "status")
            return
        parse_root = Path(self.run.state.get("parse_root", self.path("parse")))
        analysis = parse_root / result['snapshot_id'] / self.config['parse_generation']
        if "parse" not in self.run.state["steps"] and (analysis / "parse_stage_manifest.json").exists():
            failed_manifest = read(analysis / "parse_stage_manifest.json").get("status") == "failed"
            failed_gate = self.path("preindex_gate.json").exists() and read(self.path("preindex_gate.json")).get("status") == "fail"
            if failed_manifest or failed_gate:
                attempt = 2
                while self.path(f"parse_attempt{attempt}").exists():
                    attempt += 1
                parse_root = self.path(f"parse_attempt{attempt}")
                if failed_gate:
                    immutable(self.path(f"failed_gates/preindex_before_attempt{attempt}.json"), read(self.path("preindex_gate.json")))
                analysis = parse_root / result['snapshot_id'] / self.config['parse_generation']
                self.run.event("failed_parse_preserved", next_attempt=str(parse_root))
        self.run.state["parse_root"] = str(parse_root)
        self.run.state["analysis"] = str(analysis)
        self.run.save()
        def parsing():
            if not (analysis / "parse_stage_manifest.json").exists():
                self.command("scripts/parse_code_snapshot.py", result["snapshot_id"],
                    "--snapshot-root", self.config["snapshot_root"], "--staging-root", parse_root,
                    "--generation", self.config["parse_generation"],
                    "--base-generation-directory", self.config["base_analysis"])
            self.command("scripts/check_code_preindex_gate.py", result["snapshot_id"],
                "--snapshot-root", self.config["snapshot_root"], "--staging-root", parse_root,
                "--generation", self.config["parse_generation"], "--output", self.path("preindex_gate.json"))
            return {}, [*analysis.rglob("*.json"), self.path("preindex_gate.json")]
        self.run.step("parse", parsing)
        self.run.bind_tree(analysis, "*.json")
        # Older runs parsed unchanged sources again. Wall-clock resource limits
        # can select a different parser path and invalidate otherwise reusable
        # vectors. Preserve that attempt and publish a baseline-consistent stage.
        if ("prepare_index" not in self.run.state["steps"] and
                read(analysis / "parse_stage_manifest.json").get("reused_from_generation") !=
                str(Path(self.config["base_analysis"]).resolve())):
            original_analysis = analysis
            recovery_root = self.path("parse_baseline_reuse")
            recovered = recovery_root / result["snapshot_id"] / self.config["parse_generation"]
            def recover_baseline_parse():
                if not (recovered / "parse_stage_manifest.json").exists():
                    self.command("scripts/parse_code_snapshot.py", result["snapshot_id"],
                        "--snapshot-root", self.config["snapshot_root"], "--staging-root", recovery_root,
                        "--generation", self.config["parse_generation"],
                        "--reuse-directory", original_analysis,
                        "--base-generation-directory", self.config["base_analysis"])
                self.command("scripts/check_code_preindex_gate.py", result["snapshot_id"],
                    "--snapshot-root", self.config["snapshot_root"], "--staging-root", recovery_root,
                    "--generation", self.config["parse_generation"], "--output", self.path("preindex_baseline_reuse_gate.json"))
                return {"original_analysis": str(original_analysis), "analysis": str(recovered)}, [
                    *recovered.rglob("*.json"), self.path("preindex_baseline_reuse_gate.json")]
            recovery = self.run.step("parse_baseline_reuse", recover_baseline_parse)
            analysis = Path(recovery["analysis"])
            self.run.bind_tree(analysis, "*.json")
            self.run.state["analysis"] = str(analysis)
            self.run.save()
        elif "parse_baseline_reuse" in self.run.state["steps"]:
            analysis = Path(self.run.step("parse_baseline_reuse", lambda: None)["analysis"])
            self.run.state["analysis"] = str(analysis)
            self.run.save()
        def prepare_index():
            packet = build_dependency_review_packet(Path(self.config["snapshot_root"]) / result["snapshot_id"], analysis)
            def preserve_unapproved(path, model):
                if not path.exists() or read(path) == model.model_dump(mode="json"):
                    return
                if "parse_baseline_reuse" not in self.run.state["steps"] or self.path("embedding_request.json").exists():
                    raise ValueError("Prepared input changed; reconcile existing embedding approval")
                prior = self.path("failed_preparation") / f"{path.stem}-{sha(path)}.json"
                prior.parent.mkdir(parents=True, exist_ok=True)
                if prior.exists():
                    raise ValueError(f"Prior failed preparation already preserved: {prior}")
                path.replace(prior)
                self.run.event("failed_preparation_preserved", path=str(prior))
            preserve_unapproved(self.path("dependency_packet.json"), packet)
            dump_model(self.path("dependency_packet.json"), packet)
            prepared = build_code_index_artifact(analysis, embedding_model=self.config["model"])
            from app.code_indexing.contract import verify_prepared_code_index_artifact
            verify_prepared_code_index_artifact(prepared, analysis, expected_policy_sha256=prepared.analysis_policy_sha256)
            prepared_path = self.path("provisional_prepared.json")
            preserve_unapproved(prepared_path, prepared)
            dump_model(prepared_path, prepared)
            plan = plan_embeddings(prepared, [Path(self.config["base_artifact"])], delta["unchanged"],
                                   self.config["price_per_million"], self.config["pricing_basis"])
            immutable(self.path("embedding_request.json"), plan)
            return {"missing": plan["unique_missing_inputs"], "cached": plan["cached_records"]}, [
                self.path("dependency_packet.json"), self.path("provisional_prepared.json"), self.path("embedding_request.json")]
        self.run.step("prepare_index", prepare_index)
        self.run.transition("AWAITING_EMBEDDING_APPROVAL", "build")

    def stage_collection(self, artifact_path, label):
        from qdrant_client import QdrantClient
        from app.code_indexing.qdrant import index_code_artifact_new_collection, verify_code_collection
        from dataclasses import asdict
        artifact = load_code_index_artifact(artifact_path)
        run_label = self.run.state['run_id'].lower().replace('-', '_')
        stem = f"code_custom_update_{run_label}_{label}_{artifact.artifact_identity_sha256[:12]}"
        # Existing unowned collections are never overwritten. Intent precedes creation.
        intent = self.path(f"{label}_collection_intent.json")
        value = read(intent) if intent.exists() else {"collection": stem, "artifact": artifact.artifact_identity_sha256}
        immutable(intent, value)
        try:
            client = QdrantClient(path=self.config["code_store"])
        except RuntimeError as exc:
            raise RuntimeError("Local code store is occupied. Stop its serving owner and resume this action.") from exc
        try:
            if client.collection_exists(value["collection"]):
                try:
                    check = verify_code_collection(client, collection_name=value["collection"], artifact=artifact)
                except RuntimeError:
                    # Preserve an interrupted collection and record a new attempt.
                    attempt = 2
                    while client.collection_exists(f"{stem}_attempt{attempt}"):
                        attempt += 1
                    self.run.event("collection_preserved", collection=value["collection"])
                    value = {**value, "collection": f"{stem}_attempt{attempt}"}
                    write(intent, value)
                    check = index_code_artifact_new_collection(client, collection_name=value["collection"], artifact=artifact)
            else:
                check = index_code_artifact_new_collection(client, collection_name=value["collection"], artifact=artifact)
            write(self.path(f"{label}_collection_verified.json"), asdict(check))
        finally:
            client.close()
        return value["collection"]

    def evaluate(self, artifact, lineage, code_files, combined_files, prefix, preliminary=False):
        if preliminary:
            from app.code_retrieval.service import retrieve_code_evidence
            from app.fdd_code_lineage.evaluation import load_code_combined_eval_cases, build_code_combined_retrieval_case_report
            code = load_code_index_artifact(Path(artifact))
            cases = []
            for path in code_files:
                for case in load_code_combined_eval_cases(path):
                    result = retrieve_code_evidence(artifact=code, query=case.question, mode="lexical", allow_provisional=True)
                    cases.append(build_code_combined_retrieval_case_report(case=case, retrieval=result).model_dump(mode="json"))
            report = self.path(prefix + "_code_evaluation.json")
            write(report, {"status": "preliminary", "release_gate_eligible": False, "external_api_calls": 0,
                           "artifact_identity": code.artifact_identity_sha256, "cases": cases})
            return [report]
        outputs = []
        for mode, files in (("code", code_files), ("combined", combined_files)):
            if not files:
                continue
            report = self.path(f"{prefix}_{mode}_evaluation.json")
            if report.exists():
                from app.activation.code_generation import verify_report
                from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
                verify_report(report, artifact=load_code_index_artifact(Path(artifact)),
                              lineage=load_reviewed_lineage(Path(lineage)) if mode == "combined" else None)
                outputs.append(report)
                continue
            attempt = 1
            attempt_report = report.with_name(report.stem + f"_attempt{attempt}.json")
            while attempt_report.exists():
                attempt += 1
                attempt_report = report.with_name(report.stem + f"_attempt{attempt}.json")
            arguments = []
            for file in files:
                arguments.extend(["--eval-file", file])
            arguments += ["--code-artifact", artifact, "--analysis-directory", self.run.state["analysis"],
                          "--code-mode", "lexical", "--output-file", attempt_report]
            if preliminary:
                arguments += ["--allow-unreviewed"]
            if mode == "combined":
                arguments += ["--lineage-artifact", lineage, "--fdd-generation", self.config["fdd_generation"],
                              "--fdd-directory", self.config["fdd_directory"]]
            rc = self.command("scripts/run_code_combined_retrieval_eval.py", *arguments,
                              name=f"{prefix}_{mode}", tolerate_failure=True)
            if not attempt_report.exists():
                raise RuntimeError(f"Evaluator could not produce {attempt_report}; inspect logs")
            if not preliminary and (rc or not read(attempt_report)["summary"]["release_gate_eligible"]):
                raise RuntimeError(f"Final {mode} regression gate failed: {attempt_report}. Existing expectations remain unchanged.")
            # Apply the same per-case and identity requirements as promotion
            # before publishing a reusable successful checkpoint.
            from app.activation.code_generation import verify_report
            from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
            try:
                verify_report(attempt_report, artifact=load_code_index_artifact(Path(artifact)),
                              lineage=load_reviewed_lineage(Path(lineage)) if mode == "combined" else None)
            except ValueError as exc:
                raise RuntimeError(f"Final {mode} regression gate failed: {attempt_report}. {exc}") from exc
            immutable(report, read(attempt_report))
            outputs.append(report)
        return outputs

    def build(self, args):
        if self.run.state.get("pending_review_embedding"):
            pending = self.run.state["pending_review_embedding"]
            if finalized_review_identity(self.dir) != pending["review_hash"]:
                raise ValueError("Review changed; run finalize to prepare the matching additional-input request")
            plan = read(pending["request"])
            approval = Path(pending["approval"])
            requested_ceiling = getattr(args, "max_usd", None)
            if approval.exists() and requested_ceiling and positive_decimal(requested_ceiling) != positive_decimal(read(approval)["max_usd"]):
                approval = approval.parent / ("embedding-approval-" + digest(str(requested_ceiling))[:16] + ".json")
                pending["approval"] = str(approval)
                self.run.save()
            if not approval.exists():
                print(json.dumps(plan, indent=2))
                ceiling = getattr(args, "max_usd", None) or self.prompt("Maximum USD for additional reviewed inputs: ").strip()
                positive_decimal(ceiling)
                if self.prompt(f"Type EMBED {plan['request_hash']} to approve: ").strip() != f"EMBED {plan['request_hash']}":
                    raise PermissionError("Additional-input approval is required")
                immutable(approval, {"decision": "approved", "request_hash": plan["request_hash"], "max_usd": ceiling,
                    "approved_by": self.config["request"]["reviewer"], "at": now()})
            embedded = checkpointed_embed(load_code_index_artifact(Path(pending["prepared"])), plan, read(approval),
                Path(pending["batches"]), approved_client)
            dump_model(Path(pending["output"]), embedded)
            self.run.state.pop("pending_review_embedding")
            self.run.transition("REVIEWED_INPUTS_EMBEDDED", "finalize")
            return
        if self.run.state.get("final"):
            print("Build already finalized; completed state retained.")
            return
        if "prepare_index" not in self.run.state["steps"]:
            raise ValueError("Run prepare first")
        plan = read(self.path("embedding_request.json"))
        approval = Path(self.run.state.get("embedding_approval", self.path("embedding_approval.json")))
        requested_ceiling = getattr(args, "max_usd", None)
        if approval.exists() and requested_ceiling and positive_decimal(requested_ceiling) != positive_decimal(read(approval)["max_usd"]):
            if "embed" in self.run.state["steps"]:
                raise ValueError("Embedding is already complete; no further budget approval is needed")
            approval = self.path(f"approvals/embedding-{digest({'request': plan['request_hash'], 'ceiling': str(requested_ceiling)})}.json")
        if not approval.exists():
            print(json.dumps({k: v for k, v in plan.items() if k not in {"missing", "caches"}}, indent=2))
            ceiling = getattr(args, "max_usd", None) or self.prompt("Maximum USD authorized for this run: ").strip()
            positive_decimal(ceiling)
            exact = f"EMBED {plan['request_hash']}"
            if self.prompt(f"Authorize these code excerpts and cost by typing {exact}: ").strip() != exact:
                raise PermissionError("Embedding approval was not supplied")
            immutable(approval, {"request_hash": plan["request_hash"], "decision": "approved", "max_usd": str(ceiling),
                                 "approved_by": self.config["request"]["reviewer"], "at": now()})
        self.run.state["embedding_approval"] = str(approval)
        self.run.save()
        def embedding():
            artifact = load_code_index_artifact(self.path("provisional_prepared.json"))
            embedded = checkpointed_embed(artifact, plan, read(approval), self.path("batches"),
                                          approved_client)
            dump_model(self.path("provisional_embedded.json"), embedded)
            return read(self.path("batches/reuse_report.json")), [self.path("provisional_embedded.json"), approval,
                                                                 *self.path("batches").glob("*.json")]
        self.run.step("embed", embedding)
        self.run.step("stage_provisional", lambda: (
            {"collection": self.stage_collection(self.path("provisional_embedded.json"), "provisional")},
            [self.path("provisional_collection_verified.json")]))
        self.run.step("preliminary", lambda: ({}, self.evaluate(self.path("provisional_embedded.json"), None,
            self.config["code_evals"], [], "preliminary", preliminary=True)))
        self.run.step("review_packet", self.make_review)
        self.run.transition("AWAITING_SME_REVIEW", "finalize")

    def make_review(self):
        from app.fdd_code_lineage.semantic_proposals import build_proposals
        from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
        from app.fdd_code_lineage.models import _load_analysis
        from app.code_updates.review import item
        artifact = load_code_index_artifact(self.path("provisional_embedded.json"))
        report_path = self.path("proposals.json")
        if not report_path.exists():
            proposals = build_proposals(fdd_stage=Path(self.config["fdd_stage"]),
                snapshot_directory=Path(self.config["snapshot_root"]) / artifact.snapshot_id,
                analysis_directory=Path(self.run.state["analysis"]), code_artifact_path=self.path("provisional_embedded.json"),
                registry_path=Path(self.config["registry"]), allow_provisional=True,
                progress=lambda message: print(message, flush=True))
            immutable(report_path, proposals)
        proposals = read(report_path)
        items = []
        packet = read(self.path("dependency_packet.json"))
        from app.code_updates.inheritance import affected_context, carried_dependency_decisions
        from app.code_ingestion.dependency_review import build_dependency_review_packet, DependencyReviewPacket
        from app.code_ingestion.dependency_review_ledger import load_dependency_review_ledger
        delta = self.run.state["steps"]["snapshot"]["result"]["diff"]
        affected = affected_context(self.config["base_analysis"], self.run.state["analysis"],
                                    delta["added"] + delta["modified"] + delta["deleted"])
        base_ledger = load_dependency_review_ledger(Path(self.config["base_dependency_ledger"]))
        base_packet = build_dependency_review_packet(Path(self.config["snapshot_root"]) / self.run.state["base_snapshot_id"],
                                                     Path(self.config["base_analysis"]))
        carries = {}
        if base_packet.packet_identity_sha256 == base_ledger.packet_identity_sha256:
            carries = carried_dependency_decisions(base_packet, DependencyReviewPacket.model_validate(packet), base_ledger, affected)
        for case in packet["cases"]:
            carry = carries.get(case["review_id"])
            items.append(item("dependency", {**case, **({"carry_decision": carry} if carry else {})},
                "Does this dependency classification match these source occurrences?",
                "validated_carry_forward" if carry else "machine_proposal"))
        analyses = _load_analysis(Path(self.run.state["analysis"]))
        changed = set(self.run.state["steps"]["snapshot"]["result"]["diff"]["added"] +
                      self.run.state["steps"]["snapshot"]["result"]["diff"]["modified"])
        # One independently reviewed routine expectation per changed source, with
        # all inventory retained separately. Generated checks alone are not semantic proof.
        inventory = {}
        for path, symbols in sorted(analyses.items()):
            inventory[path] = [s.model_dump(mode="json") for s in symbols]
            if path not in changed or not symbols:
                continue
            symbol = next((s for s in symbols if s.occurrence_role == "implementation"), symbols[0])
            case = dict(case_id="update-code-" + digest({"path": path, "symbol": symbol.canonical_qualified_name})[:16],
                mode="code", question=f"Explain {symbol.qualified_display_name} in {path}",
                expected_code_paths=[path], expected_code_symbols=[symbol.name.display_name],
                sme_reviewed=False, review_status="draft", rationale="Generated from parser inventory; requires independent source review.")
            matching = [r for r in artifact.records if r.source_path == path and r.source_map.start_offset < symbol.source_map.end_offset
                        and symbol.source_map.start_offset < r.source_map.end_offset]
            from app.code_retrieval.service import retrieve_code_evidence
            from app.fdd_code_lineage.evaluation import CodeCombinedEvalCase, build_code_combined_retrieval_case_report
            diagnostic = build_code_combined_retrieval_case_report(case=CodeCombinedEvalCase.model_validate(case),
                retrieval=retrieve_code_evidence(artifact=artifact, query=case["question"], mode="lexical", allow_provisional=True))
            items.append(item("evaluation", {"case": case, "symbol": symbol.model_dump(mode="json"),
                "source_excerpts": [r.citation_text for r in matching[:2]],
                "preliminary_result_not_release_evidence": diagnostic.model_dump(mode="json")}, "Is this routine retrieval expectation correct?"))
        immutable(self.path("inventory.json"), inventory)
        inherited = load_reviewed_lineage(Path(self.config["base_lineage"]))
        base_artifact = load_code_index_artifact(Path(self.config["base_artifact"]))
        from app.code_updates.inheritance import source_fingerprints as _fingerprints
        from app.retrieval.lexical_search import load_retrieval_ready_documents, search_lexical_documents
        fdd_documents = load_retrieval_ready_documents(self.config["fdd_directory"])
        for mapping in inherited.mappings:
            target_paths = {t.path for t in mapping.targets}
            current_records = [r for r in artifact.records if r.source_path in target_paths][:3]
            document_units = [d for d in fdd_documents if d.document_id == mapping.fdd_document_id]
            passages = search_lexical_documents(document_units,
                " ".join(t.qualified_name or t.path for t in mapping.targets) + " " + " ".join(r.citation_text[:500] for r in current_records), limit=2)
            evidence = dict(fdd_document_id=mapping.fdd_document_id, fdd_release_label=mapping.fdd_release_label,
                targets=[t.model_dump(mode="json") for t in mapping.targets],
                prior_mapping=mapping.model_dump(mode="json"), base_lineage_identity=inherited.artifact_identity_sha256,
                current_source_excerpts=[{"source_map": r.source_map.model_dump(mode="json"), "text": r.citation_text} for r in current_records],
                candidate_context_passages=[{"id": p.point_id, "payload": p.payload} for p in passages])
            can_carry = all(t.path not in affected and _fingerprints(base_artifact, t.path) == _fingerprints(artifact, t.path)
                            and bool(_fingerprints(artifact, t.path)) for t in mapping.targets)
            if can_carry:
                evidence["carry_decision"] = {"verdict": "accepted", "rationale": mapping.rationale,
                    "correction": {}, "prior_reviewer": mapping.reviewer, "prior_mapping_id": mapping.mapping_id,
                    "prior_lineage_identity": inherited.artifact_identity_sha256}
            items.append(item("inherited_lineage", evidence,
                "Does this existing relationship still hold with the current caller/validation context?",
                "validated_carry_forward" if can_carry else "prior_human_review"))
        existing = {(m.fdd_document_id, t.path, t.qualified_name) for m in inherited.mappings for t in m.targets}
        for document in proposals["documents"]:
            release = re.search(r"(?:^|_)R(\d+)(?:_|$)", document["document_id"])
            if not release:
                continue
            for candidate in document["candidates"]:
                target = candidate["target"]
                if (document["document_id"], target["path"], target["qualified_name"]) in existing:
                    continue
                if target["path"] not in changed:
                    continue
                target = {**target, "rationale": "Candidate discovered from stored vectors and source context; SME confirmation required."}
                items.append(item("lineage", {"fdd_document_id": document["document_id"],
                    "fdd_release_label": "R" + release[1], "targets": [target], "proposal": candidate,
                    "evaluation_question": f"What FDD requirement is implemented by {target['qualified_name']} in {target['path']}?"},
                    "Does this routine implement/support this FDD passage? Defer if uncertain."))
        candidate_paths = {t["path"] for i in items if i["kind"] in {"lineage", "inherited_lineage"} for t in i["evidence"]["targets"]}
        for path in sorted(changed - candidate_paths):
            symbols = analyses.get(path, ())
            symbol = next((s for s in symbols if s.occurrence_role == "implementation"), None)
            if symbol is None:
                continue
            boundary = dict(case_id="update-boundary-" + digest(path)[:16],
                question=f"Explain {symbol.qualified_display_name} in {path} and its documentation boundary",
                expected_code_paths=[path], expected_code_symbols=[symbol.name.display_name],
                sme_reviewed=False, review_status="draft", rationale="No reviewed mapping is proposed for this source.")
            items.append(item("documentation_boundary", {"case": boundary, "source_path": path},
                "Should this return code evidence without claiming an approved FDD implementation link?"))
            break
        def source_group(entry):
            evidence = entry["evidence"]
            paths = list(evidence.get("case", {}).get("expected_code_paths", []))
            paths += [t["path"] for t in evidence.get("targets", [])]
            paths += [e["source_path"] for e in evidence.get("examples", [])]
            return (min(paths) if paths else "", entry["kind"], entry["id"])
        items = [item(i["kind"], {**i["evidence"], "snapshot_id": artifact.snapshot_id,
            "provisional_artifact_identity": artifact.artifact_identity_sha256}, i["question"], i["provenance"]) for i in items]
        items.sort(key=source_group)
        immutable(self.path("review_evidence.json"), items)
        if not self.path("review.md").exists():
            review.render(items, self.path("review.md"))
        return {"items": len(items)}, [self.path("review_evidence.json"), report_path, self.path("inventory.json")]

    def amend_review(self, args):
        """Create one narrow, evidence-bound lineage amendment for SME review."""
        from app.fdd_code_lineage.models import _load_analysis
        from app.retrieval.lexical_search import load_retrieval_ready_documents, search_lexical_documents

        if self.path("activation.json").exists():
            raise ValueError("This run was activated; new review decisions require a new update run")
        if "review_packet" not in self.run.state["steps"]:
            raise ValueError("Build the consolidated review packet before creating an amendment")
        values = {
            "fdd_document_id": getattr(args, "fdd_document_id", None),
            "source_path": getattr(args, "source_path", None),
            "qualified_name": getattr(args, "qualified_name", None),
            "symbol_kind": getattr(args, "symbol_kind", None),
            "source_marker": getattr(args, "source_marker", None),
            "evidence_query": getattr(args, "evidence_query", None),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError("Review amendment requires: " + ", ".join(missing))
        evidence_path = self.path("review_amendment_evidence.json")
        review_path = self.path("review_amendment.md")
        if evidence_path.exists() or review_path.exists():
            raise ValueError("A review amendment already exists; review it or start a new update run")
        artifact = load_code_index_artifact(self.path("provisional_embedded.json"))
        analyses = _load_analysis(Path(self.run.state["analysis"]))
        matches = [
            symbol for symbol in analyses.get(values["source_path"], ())
            if symbol.canonical_qualified_name == values["qualified_name"]
            and symbol.symbol_kind == values["symbol_kind"]
        ]
        if not matches:
            raise ValueError("Amendment target is not an exact current parsed symbol")
        documents = load_retrieval_ready_documents(self.config["fdd_directory"])
        document_units = [item for item in documents if item.document_id == values["fdd_document_id"]]
        if not document_units:
            raise ValueError("Amendment FDD document is not in the configured FDD generation")
        source = Path(self.config["snapshot_root"]) / artifact.snapshot_id / "source" / values["source_path"]
        if not source.is_file():
            raise ValueError("Amendment source is not present in the immutable snapshot")
        source_text = source.read_text(encoding="utf-8-sig", errors="replace")
        marker = str(values["source_marker"])
        marker_at = source_text.casefold().find(marker.casefold())
        if marker_at < 0:
            raise ValueError("The supplied source marker is not present in the immutable source")
        marker_line = source_text[:marker_at].count("\n") + 1
        symbol = matches[0]
        symbol_records = [
            record for record in artifact.records
            if record.source_path == values["source_path"]
            and record.source_map.start_offset < symbol.source_map.end_offset
            and symbol.source_map.start_offset < record.source_map.end_offset
        ]
        marker_context = "\n".join(source_text.splitlines()[max(0, marker_line - 2):marker_line + 2])
        passages = search_lexical_documents(document_units, values["evidence_query"], limit=2)
        if not passages:
            raise ValueError("No FDD passages matched the evidence query; refine it before creating an amendment")
        target = {
            "module_id": artifact.module_id,
            "path": values["source_path"],
            "qualified_name": values["qualified_name"],
            "symbol_kind": values["symbol_kind"],
            "selector_scope": "all_overloads",
            "rationale": "Regression-discovered exact routine candidate, bound to the immutable source marker and FDD passage for SME review.",
        }
        release = re.search(r"(?:^|_)R(\d+)(?:_|$)", values["fdd_document_id"])
        if not release:
            raise ValueError("Amendment FDD identity has no release label")
        from app.code_updates.review import item
        entry = item("lineage", {
            "snapshot_id": artifact.snapshot_id,
            "provisional_artifact_identity": artifact.artifact_identity_sha256,
            "fdd_document_id": values["fdd_document_id"],
            "fdd_release_label": "R" + release[1],
            "targets": [target],
            "evaluation_question": f"How does {values['qualified_name']} in {values['source_path']} implement or support: {values['evidence_query']}?",
            "proposal": {
                "discovery": "final_regression_gap",
                "evidence_query": values["evidence_query"],
                "source_marker": marker,
                "source_marker_line": marker_line,
                "source_marker_context": marker_context,
                "routine_source_map": symbol.source_map.model_dump(mode="json"),
                "routine_excerpts": [record.citation_text for record in symbol_records[:2]],
                "fdd_passages": [{"unit_id": passage.point_id, "payload": passage.payload} for passage in passages],
            },
        }, "Does this exact routine implement/support the cited FDD requirement? Defer if uncertain.")
        immutable(evidence_path, [entry])
        review.render([entry], review_path)
        self.run.event("review_amendment_created", item_id=entry["id"], target=values["qualified_name"])
        print(f"Review amendment created: {review_path}")

    def finalize(self, args):
        from app.code_ingestion.dependency_review import DependencyReviewPacket
        from app.code_indexing.embedding import embed_code_index_artifact
        from app.fdd_code_lineage.models import validate_lineage_artifact
        from app.retrieval.lexical_search import load_retrieval_ready_documents
        if "review_packet" not in self.run.state["steps"]:
            raise ValueError("Run build first")
        if self.path("activation.json").exists():
            if finalized_review_identity(self.dir) != self.run.state["final_review_hash"]:
                raise ValueError("This run was activated; changed reviews require a new update run")
            print("Finalization already activated; current state retained.")
            return
        primary_items = read(self.path("review_evidence.json"))
        amendment_evidence = self.path("review_amendment_evidence.json")
        amendment_review = self.path("review_amendment.md")
        amendment_items = read(amendment_evidence) if amendment_evidence.exists() else ()
        if amendment_items and not amendment_review.exists():
            raise ValueError("Review amendment evidence exists but its Markdown packet is missing")
        items, selected = review.collect_review_decisions(
            primary_items, self.path("review.md"), amendment_items=amendment_items,
            amendment_path=amendment_review if amendment_items else None,
        )
        review_hash = finalized_review_identity(self.dir)
        revision = review_hash[:16]
        directory = self.path(f"final/{revision}")
        directory.mkdir(parents=True, exist_ok=True)
        def finalized():
            packet = DependencyReviewPacket.model_validate(read(self.path("dependency_packet.json")))
            ledger = review.dependency_ledger(packet, self.path("dependency_packet.json"), review_hash, items, selected,
                                              self.config["request"]["reviewer"])
            ledger_path = directory / "dependency_ledger.json"
            dump_model(ledger_path, ledger)
            prepared = build_code_index_artifact(Path(self.run.state["analysis"]), embedding_model=self.config["model"],
                                                 dependency_review_ledger=ledger)
            code_path = directory / "code_index_artifact.json"
            extra_plan = plan_embeddings(prepared, [self.path("provisional_embedded.json")], [],
                                        self.config["price_per_million"], self.config["pricing_basis"])
            if extra_plan["unique_missing_inputs"] and not code_path.exists():
                dump_model(directory / "prepared.json", prepared)
                immutable(directory / "embedding_request.json", extra_plan)
                self.run.state["pending_review_embedding"] = {
                    "review_hash": review_hash, "prepared": str(directory / "prepared.json"),
                    "request": str(directory / "embedding_request.json"), "approval": str(directory / "embedding_approval.json"),
                    "batches": str(directory / "batches"), "output": str(code_path)}
                self.run.save()
                raise EmbeddingApprovalRequired()
            class NoPaidClient:
                @property
                def embeddings(self):
                    raise PermissionError("Review changed embedding text. New prepare/embedding approval is required.")
            if code_path.exists():
                embedded = load_code_index_artifact(code_path)
                if embedded.artifact_identity_sha256 != prepared.artifact_identity_sha256:
                    raise ValueError("Reviewed embedding artifact belongs to different inputs")
            else:
                embedded, _ = embed_code_index_artifact(prepared, client=NoPaidClient(),
                    cache_artifact_paths=[self.path("provisional_embedded.json")])
            dump_model(code_path, embedded)
            target_overrides, path_rebindings = review.normalize_lineage_targets(items, selected, embedded)
            candidate, lineage, deferred = review.finalize_lineage(items, selected, embedded,
                self.config["fdd_generation"], review_hash, self.config["request"]["reviewer"],
                target_overrides=target_overrides)
            known = {d.document_id for d in load_retrieval_ready_documents(self.config["fdd_directory"])}
            validate_lineage_artifact(lineage, fdd_document_ids=known, code_artifact=embedded,
                                      analysis_directory=Path(self.run.state["analysis"]))
            dump_model(directory / "candidate_lineage.json", candidate)
            dump_model(directory / "reviewed_lineage.json", lineage)
            immutable(directory / "decisions.json", {"review_sha256": review_hash, "reviewer": self.config["request"]["reviewer"],
                                                      "decisions": selected, "automatic_path_rebindings": path_rebindings})
            immutable(directory / "deferred_lineage.json", deferred)
            write(self.path("deferred_lineage.json"), deferred)
            cases = review.reviewed_cases(items, selected)
            case_path = directory / "new_code_cases.jsonl"
            content = "".join(c.model_dump_json() + "\n" for c in cases)
            if case_path.exists() and case_path.read_text(encoding="utf-8") != content:
                raise ValueError("Final evaluation manifest changed")
            atomic_text(case_path, content)
            from app.fdd_code_lineage.evaluation import CodeCombinedEvalCase
            combined_cases = []
            for entry in items:
                if entry["kind"] != "lineage" or selected[entry["id"]]["verdict"] == "deferred":
                    continue
                decision, data = selected[entry["id"]], entry["evidence"]
                correction = decision["correction"]
                targets = target_overrides.get(entry["id"], correction.get("targets", data["targets"]))
                combined_cases.append(CodeCombinedEvalCase(case_id="update-combined-" + entry["id"][:16], mode="combined",
                    question=correction.get("evaluation_question", data["evaluation_question"]),
                    expected_code_paths=tuple(sorted({t["path"] for t in targets})),
                    expected_fdd_document_ids=(correction.get("fdd_document_id", data["fdd_document_id"]),),
                    require_reviewed_lineage=True, sme_reviewed=True, review_status="reviewed", rationale=decision["rationale"]))
            combined_path = directory / "new_combined_cases.jsonl"
            atomic_text(combined_path, "".join(c.model_dump_json() + "\n" for c in combined_cases))
            from app.fdd_code_lineage.documentation_boundary import DocumentationBoundaryCase, load_documentation_boundary_cases
            boundaries = {}
            for prior in self.config.get("boundary_evals", []):
                for case in load_documentation_boundary_cases(Path(prior)):
                    boundaries[case.case_id] = case
            for entry in items:
                if entry["kind"] == "documentation_boundary":
                    decision = selected[entry["id"]]
                    payload = {**entry["evidence"]["case"], **decision["correction"], "sme_reviewed": True,
                        "review_status": "reviewed", "rationale": decision["rationale"]}
                    if payload["case_id"] != entry["evidence"]["case"]["case_id"]:
                        raise ValueError("Boundary case identity cannot change")
                    case = DocumentationBoundaryCase.model_validate(payload)
                    if case.case_id in boundaries and boundaries[case.case_id] != case:
                        raise ValueError("Boundary case conflicts with an established regression")
                    boundaries[case.case_id] = case
            boundary_path = directory / "boundary_cases.jsonl"
            atomic_text(boundary_path, "".join(c.model_dump_json() + "\n" for c in boundaries.values()))
            return {"code": str(code_path), "lineage": str(directory / "reviewed_lineage.json"),
                    "ledger": str(ledger_path), "new_cases": str(case_path) if cases else None,
                    "boundary_cases": str(boundary_path) if boundaries else None,
                    "new_combined_cases": str(combined_path) if combined_cases else None}, [p for p in directory.rglob("*") if p.is_file()]
        try:
            final = self.run.step("finalize_" + revision, finalized)
        except EmbeddingApprovalRequired:
            self.run.transition("AWAITING_EMBEDDING_APPROVAL", "build")
            print("Reviewed text introduced new cache misses. Only additional inputs need approval; no paid call was made.")
            return
        self.run.state["final"] = final
        self.run.state["final_review_hash"] = review_hash
        self.run.save()
        label = "final_" + revision
        collection = self.run.step(label + "_collection", lambda: (
            {"collection": self.stage_collection(Path(final["code"]), label)}, [self.path(f"{label}_collection_verified.json")]))["collection"]
        self.run.state["final_collection"] = collection
        self.run.save()
        from app.activation.code_generation import runtime_files
        runtime_revision = digest(runtime_files(self.root))[:12]
        label += "_" + runtime_revision
        code_files = self.config["code_evals"] + ([final["new_cases"]] if final["new_cases"] else [])
        reports = self.run.step(label + "_evaluations", lambda: ({}, self.evaluate(final["code"], final["lineage"],
            code_files, self.config["combined_evals"] + ([final["new_combined_cases"]] if final["new_combined_cases"] else []), label)))
        if final["boundary_cases"]:
            self.run.step(label + "_boundary", lambda: self.boundary_eval(final, label))
        # MCP execution and documentation-boundary checks use final reviewed metadata.
        self.run.step(label + "_mcp", lambda: self.local_uat(final, collection, label))
        if digest(runtime_files(self.root))[:12] != runtime_revision:
            raise ValueError("Runtime changed while final gates ran; repeat finalize for fresh evidence")
        if self.config.get("coordinated_release"):
            # The parent knowledge workflow owns the one joint release request.
            # Do not create a code-only request that would become incompatible
            # the moment the staged FDD generation is selected.
            self.run.state["joint_ready"] = {"code": final["code"], "lineage": final["lineage"],
                                               "collection": collection, "runtime_revision": runtime_revision}
            self.run.save()
            self.run.transition("READY_FOR_PROMOTION", "status")
            return
        def promotion():
            from app.activation.code_generation import prepare, verify_request
            published_request = self.root / "data/exports/activation" / f"code-update-{self.run.state['run_id']}-{revision}-{runtime_revision}-request.json"
            if published_request.exists():
                request = read(published_request)
                verify_request(request, self.root)
                if request["target_configuration"]["CODE_QDRANT_COLLECTION_NAME"] != collection:
                    raise ValueError("Existing promotion request selects another collection")
                return {"request": str(published_request), "hash": request["request_identity_sha256"]}, [published_request]
            request = prepare(root=self.root, settings=Settings(), code_artifact=Path(final["code"]),
                analysis=Path(self.run.state["analysis"]), lineage_path=Path(final["lineage"]),
                dependency_ledger=Path(final["ledger"]), code_report=self.path(label + "_code_evaluation.json"),
                combined_report=self.path(label + "_combined_evaluation.json"), collection=collection,
                requested_by=self.config["request"]["reviewer"])
            from app.activation.code_generation import file_set, digest as promotion_digest
            review_files = [self.path("review_evidence.json"), self.path("review.md")]
            if self.path("review_amendment_evidence.json").exists():
                review_files.extend([self.path("review_amendment_evidence.json"), self.path("review_amendment.md")])
            request["evidence_files"].update(file_set(self.root, [*review_files,
                directory / "decisions.json", directory / "deferred_lineage.json", self.path(label + "_mcp_uat.json"),
                self.path("batches/reuse_report.json")]))
            if final["boundary_cases"]:
                request["evidence_files"].update(file_set(self.root, [Path(final["boundary_cases"]), self.path(label + "_boundary.json")]))
            request["request_identity_sha256"] = promotion_digest({k: v for k, v in request.items() if k != "request_identity_sha256"})
            immutable(published_request, request)
            return {"request": str(published_request), "hash": request["request_identity_sha256"]}, [published_request]
        promoted = self.run.step(label + "_promotion", promotion)
        self.run.state.update(promotion_request=promoted["request"], promotion_hash=promoted["hash"])
        self.run.transition("READY_FOR_PROMOTION", "activate")

    def boundary_eval(self, final, label):
        from scripts.run_code_documentation_boundary_eval import main
        output = self.path(label + "_boundary.json")
        attempt = 1
        candidate = self.path(label + f"_boundary_attempt{attempt}.json")
        while candidate.exists():
            attempt += 1
            candidate = self.path(label + f"_boundary_attempt{attempt}.json")
        if not output.exists():
            rc = main(["--eval-file", final["boundary_cases"], "--code-artifact", final["code"],
                "--analysis-directory", self.run.state["analysis"], "--fdd-generation", self.config["fdd_generation"],
                "--fdd-directory", self.config["fdd_directory"], "--lineage-artifact", final["lineage"], "--output-file", str(candidate)])
            if rc:
                raise RuntimeError(f"Documentation-boundary gate failed: {candidate}")
            immutable(output, read(candidate))
        value = read(output)
        if not value["summary"]["passed"] or not value["summary"]["reviewed_manifest"]:
            raise ValueError("Documentation-boundary report is not a reviewed pass")
        if value["metadata"]["eval_file_sha256"] != sha(final["boundary_cases"]):
            raise ValueError("Documentation-boundary expectations changed after evaluation")
        code = load_code_index_artifact(Path(final["code"]))
        from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
        if value["metadata"]["code_artifact_identity_sha256"] != code.artifact_identity_sha256 or value["metadata"]["lineage_artifact_identity_sha256"] != load_reviewed_lineage(Path(final["lineage"])).artifact_identity_sha256:
            raise ValueError("Documentation-boundary report references stale artifacts")
        return {"passed": True}, [output]

    def local_uat(self, final, collection, label):
        from app.code_updates.runtime import staged_uat
        output = self.path(label + "_mcp_uat.json")
        result = read(output) if output.exists() else staged_uat(self.root, self.config, self.run.state, final, collection)
        from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
        if (not result.get("passed") or result.get("collection") != collection
                or result.get("code_artifact_identity") != load_code_index_artifact(Path(final["code"])).artifact_identity_sha256
                or result.get("lineage_identity") != load_reviewed_lineage(Path(final["lineage"])).artifact_identity_sha256):
            raise ValueError("MCP UAT evidence does not match final artifacts")
        immutable(output, result)
        return {"passed": result["passed"]}, [output]

    def activate(self, args):
        from app.activation.code_generation import verify_request, switch
        from app.activation.code_modes import build_activation_approval, ActivationApproval
        from types import SimpleNamespace
        if self.run.state["status"] == "COMPLETE":
            print("This run is already complete; no configuration was changed.")
            return
        request_path = self.run.state.get("promotion_request")
        if not request_path:
            raise ValueError("Finalize successfully before activation")
        request = read(request_path)
        verify_request(request, self.root)
        if finalized_review_identity(self.dir) != self.run.state["final_review_hash"]:
            raise ValueError("Review changed after finalization")
        approval_path = Path(request_path).with_name(Path(request_path).stem + "-approval.json")
        if not approval_path.exists():
            exact = f"ACTIVATE {request['request_identity_sha256']}"
            if self.prompt(f"Approve promotion and disabled rollback by typing {exact}: ").strip() != exact:
                raise PermissionError("Exact promotion approval is required")
            approval = build_activation_approval(request=SimpleNamespace(request_identity_sha256=request["request_identity_sha256"]),
                approved_by=self.config["request"]["reviewer"], paid_smoke_authorized=False,
                internal_evidence_disclosure_authorized=False)
            dump_model(approval_path, approval)
        approval = ActivationApproval.model_validate(read(approval_path))
        self.run.state["promotion_approval"] = str(approval_path)
        self.run.save()
        if not getattr(args, "services_stopped", False):
            raise ValueError("Stop FastAPI/Desktop MCP then repeat activate with -ServicesStopped")
        from app.code_updates.runtime import stopped_mcp_guard
        with stopped_mcp_guard():
            self.apply_activation(request, approval)

    def apply_activation(self, request, approval):
        from app.activation.code_generation import switch
        receipt = self.path("activation.json")
        if not receipt.exists():
            if sha(self.root / ".env") == request["target_env_sha256"]:
                matches = [read(p) for p in (self.root / "data/exports/activation").glob("*.result.json")]
                if not any(r.get("applied") and r.get("request_identity_sha256") == request["request_identity_sha256"] for r in matches):
                    raise ValueError("Target configuration has no applied promotion receipt")
                result = {"applied": True, "recovered_from_promotion_receipt": True}
            else:
                result = switch(root=self.root, request=request, approval=approval, action="activate", settings=Settings(), apply=True)
            immutable(receipt, {**result, "at": now()})
        from app.code_updates.runtime import create_restart_challenge
        create_restart_challenge(self.run, request)
        self.run.transition("AWAITING_RESTART", "verify")

    def verify(self, args):
        from app.code_updates.runtime import verify_restart_receipt
        if not self.path("activation.json").exists():
            raise ValueError("Activate the exact approved generation before restart verification")
        receipt = getattr(args, "runtime_receipt", None) or self.run.state.get("runtime_receipt")
        if not receipt:
            valid = []
            for candidate in self.path("runtime_receipts").glob("*.json"):
                try:
                    verify_restart_receipt(self.run, candidate)
                    valid.append(candidate)
                except ValueError:
                    continue
            if len(valid) == 1:
                receipt = valid[0]
            elif len(valid) > 1:
                raise ValueError("Multiple live MCP restart receipts exist; reconcile the serving owner")
        if not receipt:
            raise ValueError("No matching live restart receipt. Restart Desktop MCP then repeat verify. For diagnostics call runtime_status and pass its JSON with -RuntimeReceipt.")
        result = verify_restart_receipt(self.run, Path(receipt))
        self.run.state["runtime_receipt"] = str(Path(receipt).resolve())
        def record_restart():
            if not self.path("restart_verified.json").exists():
                immutable(self.path("restart_verified.json"), result)
            return {"passed": True}, [self.path("restart_verified.json"), Path(receipt), self.path("activation.json")]
        self.run.step("restart_verification", record_restart)
        self.run.transition("COMPLETE", "status")

    def execute(self, action, args):
        if action != "init":
            if not self.config:
                raise ValueError("Run init first")
            self.run.check_bindings()
            if action in {"prepare", "build", "finalize"} and not self.path("activation.json").exists():
                current = Settings()
                if Path(current.code_index_artifact_path).resolve() != Path(self.config["base_artifact"]).resolve() or current.fdd_generation != self.config["fdd_generation"]:
                    raise ValueError("Active baseline changed since init. Start a new run against the new baseline.")
            load_snapshot_manifest(Path(self.config["snapshot_root"]) / self.config["request"]["base_snapshot_id"])
            if self.run.state.get("snapshot_id"):
                load_snapshot_manifest(Path(self.config["snapshot_root"]) / self.run.state["snapshot_id"])
            # Recheck every finished stage before spending, reviewing or promoting.
            for entry in self.run.state["steps"].values():
                for path, expected in entry["outputs"].items():
                    if not Path(path).is_file() or sha(path) != expected:
                        raise ValueError(f"Checkpoint output changed: {path}")
        getattr(self, action.replace("-", "_"))(args)
