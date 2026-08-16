from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_ingestion.plsql_models import CodeParseStageManifest, CodeRetrievalArtifact


CODE_INDEX_CONTRACT_DIRECTORY = "code_index_contract_v2"
CODE_EMBEDDING_INPUT_VERSION = "code_embedding_input_v1"


def build_code_index_artifact(
    parse_stage_directory: Path,
    *,
    embedding_model: str,
) -> CodeIndexArtifact:
    manifest = CodeParseStageManifest.model_validate_json(
        (parse_stage_directory / "parse_stage_manifest.json").read_text(encoding="utf-8")
    )
    records: list[CodeIndexRecord] = []
    module_id: str | None = None
    for relative_path in manifest.retrieval_artifacts:
        retrieval = CodeRetrievalArtifact.model_validate_json(
            (parse_stage_directory / relative_path).read_text(encoding="utf-8")
        )
        for unit in retrieval.units:
            resolved_module = _module_from_snapshot(manifest.snapshot_id)
            module_id = module_id or resolved_module
            content_hash = _sha256(unit.retrieval_text)
            cache_key = _cache_key(content_hash, embedding_model)
            records.append(
                CodeIndexRecord(
                    unit_id=unit.unit_id,
                    point_id=build_code_point_id(manifest.snapshot_id, unit.unit_id),
                    unit_index=len(records),
                    snapshot_id=manifest.snapshot_id,
                    module_id=resolved_module,
                    source_path=unit.source_path,
                    source_kind=unit.source_kind,
                    display_name=unit.display_name,
                    package_name=unit.package_name,
                    source_map=unit.source_map,
                    parent_unit_id=unit.parent_unit_id,
                    parent_source_map=unit.parent_source_map,
                    chunk_index=unit.chunk_index,
                    chunk_count=unit.chunk_count,
                    parser_state=unit.parser_state,
                    conditional_state=unit.conditional_state,
                    citation_text=unit.text,
                    embedding_text=unit.retrieval_text,
                    content_sha256=content_hash,
                    cache_key=cache_key,
                    embedding_model=embedding_model,
                )
            )
    ordered = tuple(sorted(records, key=lambda record: (record.source_path.casefold(), record.source_map.start_offset, record.unit_id)))
    ordered = tuple(record.model_copy(update={"unit_index": index}) for index, record in enumerate(ordered))
    identity = _artifact_identity(
        snapshot_id=manifest.snapshot_id,
        snapshot_hash=manifest.snapshot_content_sha256,
        parse_generation=manifest.parser_generation,
        policy_hash=manifest.analysis_policy_sha256,
        dependency_review_status="draft",
        embedding_model=embedding_model,
        records=ordered,
    )
    return CodeIndexArtifact(
        status="prepared",
        snapshot_id=manifest.snapshot_id,
        snapshot_content_sha256=manifest.snapshot_content_sha256,
        parse_generation=manifest.parser_generation,
        analysis_policy_sha256=manifest.analysis_policy_sha256,
        dependency_review_status="draft",
        module_id=module_id or _module_from_snapshot(manifest.snapshot_id),
        embedding_model=embedding_model,
        total_records=len(ordered),
        artifact_identity_sha256=identity,
        records=ordered,
    )


def build_code_point_id(snapshot_id: str, unit_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"code-point-v1|{snapshot_id}|{unit_id}"))


def write_code_index_artifact_no_overwrite(
    artifact: CodeIndexArtifact,
    target_directory: Path,
) -> Path:
    if target_directory.exists():
        raise FileExistsError(f"Code index generation already exists: {target_directory}")
    target_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_directory.parent / f".{target_directory.name}.tmp"
    if temporary.exists():
        raise FileExistsError(f"Temporary code index target already exists: {temporary}")
    temporary.mkdir()
    try:
        output = temporary / "code_index_artifact.json"
        output.write_text(
            json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        observed = CodeIndexArtifact.model_validate_json(output.read_text(encoding="utf-8"))
        if observed != artifact:
            raise RuntimeError("Persisted code index artifact failed round-trip validation")
        temporary.replace(target_directory)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target_directory / "code_index_artifact.json"


def load_code_index_artifact(path: Path) -> CodeIndexArtifact:
    return CodeIndexArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def code_index_generation_name(embedding_model: str) -> str:
    safe_model = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    return f"code_index_{safe_model}_v1"


def _module_from_snapshot(snapshot_id: str) -> str:
    marker = "-r"
    return snapshot_id.rsplit(marker, 1)[0] if marker in snapshot_id else snapshot_id


def _cache_key(content_hash: str, embedding_model: str) -> str:
    return _sha256(
        json.dumps(
            {
                "content_sha256": content_hash,
                "embedding_input_version": CODE_EMBEDDING_INPUT_VERSION,
                "embedding_model": embedding_model,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _artifact_identity(**values) -> str:
    records = values.pop("records")
    payload = {
        **values,
        "records": [
            {
                "unit_id": record.unit_id,
                "point_id": record.point_id,
                "content_sha256": record.content_sha256,
                "cache_key": record.cache_key,
                "source_path": record.source_path,
                "source_map": record.source_map.model_dump(mode="json"),
            }
            for record in records
        ],
    }
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
