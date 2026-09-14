"""Paid requests are bounded by immutable inputs and durable per-batch intents."""
from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
import tiktoken

from app.code_indexing.embedding import _group_records, _load_cache, _ordered_response_items
from app.code_updates.storage import digest, immutable, read, sha, write


def approved_client():
    from app.core.config import get_settings
    from app.embeddings.client import get_embedding_client
    if (get_settings().openai_base_url or "https://api.openai.com/v1").rstrip("/") != "https://api.openai.com/v1":
        raise PermissionError("This workflow authorizes the OpenAI API endpoint only; reconcile the configured endpoint before sending code")
    return get_embedding_client(max_retries=0)


def positive_decimal(value):
    number = Decimal(str(value))
    if not number.is_finite() or number <= 0:
        raise ValueError("A positive finite USD amount is required")
    return number


def plan_embeddings(artifact, caches, unchanged_paths, price, pricing_basis):
    rate = positive_decimal(price)
    cache = _load_cache(caches, embedding_model=artifact.embedding_model)
    groups = _group_records(artifact)
    missing = []
    for key, records in groups.items():
        if key in cache:
            continue
        paths = sorted({r.source_path for r in records})
        if set(paths) & set(unchanged_paths):
            raise ValueError(f"Unexpected embedding miss in unchanged source: {paths}")
        tokens = len(tiktoken.get_encoding("cl100k_base").encode(records[0].embedding_text, disallowed_special=()))
        if not 1 <= tokens <= 8191:
            raise ValueError(f"Embedding input exceeds token limit: {paths}. Rechunk before approval.")
        missing.append({"cache_key": key, "text_sha256": records[0].content_sha256,
                        "source_paths": paths, "token_upper_bound": tokens})
    token_bound = sum(i["token_upper_bound"] for i in missing)
    value = {"schema_version": "code_update_embedding_request_v1",
             "artifact_identity": artifact.artifact_identity_sha256,
             "model": artifact.embedding_model, "input_version": artifact.embedding_input_version,
             "endpoint": "https://api.openai.com/v1",
             "caches": {str(Path(p).resolve()): sha(p) for p in caches},
             "missing": missing, "cached_records": sum(r.cache_key in cache for r in artifact.records),
             "records": len(artifact.records), "unique_missing_inputs": len(missing),
             "token_count_method": "cl100k_base", "tokenizer_version": tiktoken.__version__,
             "token_upper_bound": token_bound, "usd_per_million_tokens": str(rate),
             "pricing_basis": pricing_basis, "cost_upper_bound_usd": str(rate * token_bound / 1_000_000),
             "max_retries": 0, "batch_size": 32}
    return {**value, "request_hash": digest(value)}


def validate_plan(plan, artifact):
    if digest({k: v for k, v in plan.items() if k != "request_hash"}) != plan["request_hash"]:
        raise ValueError("Embedding request identity mismatch")
    if not 1 <= plan["batch_size"] <= 32 or plan["max_retries"] != 0 or plan["endpoint"] != "https://api.openai.com/v1":
        raise ValueError("Embedding request exceeds the fixed request/retry/endpoint policy")
    if artifact.artifact_identity_sha256 != plan["artifact_identity"]:
        raise ValueError("Embedding request belongs to different prepared content")
    if plan["model"] != artifact.embedding_model or plan["input_version"] != artifact.embedding_input_version or plan["tokenizer_version"] != tiktoken.__version__:
        raise ValueError("Approved embedding model/tokenizer changed; a new request is required")
    for path, expected in plan["caches"].items():
        if sha(path) != expected:
            raise ValueError("Approved embedding cache changed")
    groups = _group_records(artifact)
    cache = _load_cache([Path(p) for p in plan["caches"]], embedding_model=artifact.embedding_model)
    expected_keys = set(groups) - set(cache)
    if {item["cache_key"] for item in plan["missing"]} != expected_keys:
        raise ValueError("Approved embedding inputs differ from actual cache misses")
    for item in plan["missing"]:
        if not 1 <= item["token_upper_bound"] <= 8191:
            raise ValueError("Embedding input token limit exceeded")
        record = groups[item["cache_key"]][0]
        if item["text_sha256"] != record.content_sha256 or item["token_upper_bound"] != len(tiktoken.get_encoding("cl100k_base").encode(record.embedding_text, disallowed_special=())):
            raise ValueError("Approved embedding text changed")


def checkpointed_embed(artifact, plan, approval, directory, client_factory):
    validate_plan(plan, artifact)
    if approval.get("request_hash") != plan["request_hash"] or approval.get("decision") != "approved":
        raise PermissionError("Exact embedding request approval is required")
    budget = positive_decimal(approval["max_usd"])
    rate = positive_decimal(plan["usd_per_million_tokens"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    groups = _group_records(artifact)
    cache = _load_cache([Path(p) for p in plan["caches"]], embedding_model=artifact.embedding_model)
    vectors = {key: item.vector for key, item in cache.items() if key in groups}
    committed = Decimal(0)
    requests = 0
    client = None
    for offset in range(0, len(plan["missing"]), plan["batch_size"]):
        batch = plan["missing"][offset:offset + plan["batch_size"]]
        batch_id = digest({"request": plan["request_hash"], "inputs": batch})
        intent_path = directory / f"{offset:08d}.intent.json"
        result_path = directory / f"{offset:08d}.result.json"
        cost = sum(i["token_upper_bound"] for i in batch) * rate / 1_000_000
        if committed + cost > budget:
            raise PermissionError("Embedding budget would be exceeded before next batch")
        committed += cost
        if result_path.exists():
            result = read(result_path)
            if not intent_path.exists() or read(intent_path)["batch_id"] != batch_id:
                raise ValueError("Embedding receipt has no matching intent")
            if result["batch_id"] != batch_id or digest(result["vectors"]) != result["vectors_sha256"]:
                raise ValueError("Embedding batch receipt changed")
        else:
            if intent_path.exists():
                raise RuntimeError(f"Uncertain paid request outcome: {intent_path}. Do not resend; reconcile provider records.")
            if client is None:
                client = client_factory()
            immutable(intent_path, {"batch_id": batch_id, "request_hash": plan["request_hash"],
                                    "committed_cost_usd": str(cost), "inputs": batch})
            response = client.embeddings.create(model=plan["model"],
                input=[groups[item["cache_key"]][0].embedding_text for item in batch])
            items = _ordered_response_items(response.data, expected_count=len(batch))
            received = {item["cache_key"]: list(answer.embedding) for item, answer in zip(batch, items, strict=True)}
            validate_vectors(received, {i["cache_key"] for i in batch})
            result = {"batch_id": batch_id, "vectors": received, "vectors_sha256": digest(received),
                      "provider_request_id": getattr(response, "_request_id", None),
                      "usage": response.usage.model_dump() if getattr(response, "usage", None) else None}
            immutable(result_path, result)
            requests += 1
        validate_vectors(result["vectors"], {i["cache_key"] for i in batch})
        vectors.update(result["vectors"])
    validate_vectors(vectors, set(groups))
    records = tuple(record.model_copy(update={"vector": tuple(vectors[record.cache_key]),
                    "embedding_status": "cached" if record.cache_key in cache else "embedded"}) for record in artifact.records)
    embedded = artifact.model_copy(update={"records": records, "status": "embedded", "vector_dimension": 3072})
    # Validate the reconstructed model, not just the unchecked model_copy.
    embedded = type(artifact).model_validate(embedded.model_dump())
    summary = {"cached_records": sum(r.cache_key in cache for r in records),
               "embedded_records": sum(r.cache_key not in cache for r in records),
               "new_requests_this_invocation": requests, "committed_cost_upper_bound_usd": str(committed),
               "validated_batch_count": len(list(directory.glob("*.result.json"))),
               "prepared_artifact_identity": artifact.artifact_identity_sha256,
               "cache_files": plan["caches"],
               "cache_artifact_identities": {p: read(p)["artifact_identity_sha256"] for p in plan["caches"]},
               "per_path": {path: {"cached": sum(r.source_path == path and r.cache_key in cache for r in records),
                                   "embedded": sum(r.source_path == path and r.cache_key not in cache for r in records)}
                            for path in sorted({r.source_path for r in records})}}
    write(directory / "reuse_report.json", summary)
    return embedded


def validate_vectors(vectors, expected):
    if set(vectors) != expected:
        raise ValueError("Embedding response keys do not match approved batch")
    if any(len(v) != 3072 or not all(math.isfinite(x) for x in v) for v in vectors.values()):
        raise ValueError("Embedding vectors must be finite, 3072 dimensional")
