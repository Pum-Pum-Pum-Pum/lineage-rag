from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.schemas.readiness_api import ReadinessCheck, ReadinessResponse
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client
from app.code_indexing.contract import load_code_index_artifact


router = APIRouter(tags=["readiness"])
logger = get_logger("readiness_api")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def readiness_check(
    knowledge_mode: Literal["fdd", "code", "combined"] = "fdd",
) -> ReadinessResponse:
    """Return whether the backend dependencies are ready for the active mode.

    Unlike `/health`, this endpoint performs local dependency/artifact checks.
    It still intentionally avoids embeddings, LLM generation, and retrieval.
    """

    settings = get_settings()

    if knowledge_mode != "fdd":
        return _extended_mode_readiness(settings, knowledge_mode)

    try:
        retrieval_config = build_retrieval_runtime_config(settings)
    except ValueError as exc:
        logger.exception("Invalid retrieval runtime configuration during readiness check")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid retrieval runtime configuration.",
        ) from exc

    retrieval_mode = retrieval_config.retrieval_mode
    qdrant_required = _requires_qdrant_collection(retrieval_mode)
    lexical_required = _requires_lexical_artifacts(retrieval_mode)

    checks: list[ReadinessCheck] = [
        ReadinessCheck(
            name="retrieval_config",
            required=True,
            is_ready=True,
            detail="Retrieval runtime configuration is valid.",
        ),
        _check_model_configuration(settings),
        _check_retrieval_ready_artifacts(settings.processed_dir, required=lexical_required),
    ]

    client = None
    try:
        if qdrant_required:
            client = create_persistent_qdrant_client(settings.qdrant_local_path)
            collection_exists = client.collection_exists(settings.qdrant_collection_name)
            checks.append(
                ReadinessCheck(
                    name="qdrant_collection",
                    required=True,
                    is_ready=collection_exists,
                    detail=(
                        "Required Qdrant collection exists."
                        if collection_exists
                        else "Required Qdrant collection is missing. Run indexing before querying."
                    ),
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    name="qdrant_collection",
                    required=False,
                    is_ready=True,
                    detail="Qdrant collection is not required for lexical-only retrieval.",
                )
            )
    except Exception as exc:
        logger.exception("Qdrant readiness check failed")
        checks.append(
            ReadinessCheck(
                name="qdrant_collection",
                required=qdrant_required,
                is_ready=False,
                detail="Qdrant readiness check failed.",
            )
        )
    finally:
        if client is not None:
            client.close()

    response = _build_response(
        app_name=settings.app_name,
        environment=settings.environment,
        retrieval_mode=retrieval_mode,
        qdrant_required=qdrant_required,
        lexical_required=lexical_required,
        checks=checks,
    )

    if not response.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )

    return response


def _extended_mode_readiness(settings, knowledge_mode: str) -> ReadinessResponse:
    checks = [_check_model_configuration(settings)]
    enabled = bool(getattr(settings, "code_modes_enabled", False))
    checks.append(
        ReadinessCheck(
            name="code_modes_activation",
            required=True,
            is_ready=enabled,
            detail=(
                "Code/combined modes are activated."
                if enabled
                else "Code/combined modes are disabled by configuration."
            ),
        )
    )
    try:
        artifact = load_code_index_artifact(settings.code_index_artifact_path)
        artifact_ready = artifact.status == "embedded" and artifact.dependency_review_status == "reviewed"
    except Exception:
        artifact_ready = False
        artifact = None
    checks.append(
        ReadinessCheck(
            name="code_artifact",
            required=True,
            is_ready=artifact_ready,
            detail="Reviewed embedded code artifact is available." if artifact_ready else "Code artifact is missing or invalid.",
        )
    )
    code_client = None
    try:
        code_client = create_persistent_qdrant_client(settings.code_qdrant_local_path)
        exists = code_client.collection_exists(settings.code_qdrant_collection_name)
        count_matches = bool(
            exists
            and artifact is not None
            and code_client.get_collection(settings.code_qdrant_collection_name).points_count
            == artifact.total_records
        )
        checks.append(
            ReadinessCheck(
                name="code_qdrant_generation",
                required=True,
                is_ready=count_matches,
                detail="Code collection exists with the exact artifact count." if count_matches else "Code collection is missing or does not match the artifact count.",
            )
        )
    except Exception:
        checks.append(ReadinessCheck(name="code_qdrant_generation", required=True, is_ready=False, detail="Code collection readiness failed."))
    finally:
        if code_client is not None:
            code_client.close()
    if knowledge_mode == "combined":
        checks.append(_check_retrieval_ready_artifacts(settings.processed_dir, required=True))
        lineage_path = Path(settings.fdd_code_lineage_artifact_path)
        checks.append(
            ReadinessCheck(
                name="reviewed_lineage_artifact",
                required=True,
                is_ready=lineage_path.is_file(),
                detail="Reviewed lineage artifact exists." if lineage_path.is_file() else "Reviewed lineage artifact is missing.",
            )
        )
        fdd_client = None
        try:
            fdd_client = create_persistent_qdrant_client(settings.qdrant_local_path)
            exists = fdd_client.collection_exists(settings.qdrant_collection_name)
            checks.append(ReadinessCheck(name="fdd_qdrant_generation", required=True, is_ready=exists, detail="FDD collection exists." if exists else "FDD collection is missing."))
        except Exception:
            checks.append(ReadinessCheck(name="fdd_qdrant_generation", required=True, is_ready=False, detail="FDD collection readiness failed."))
        finally:
            if fdd_client is not None:
                fdd_client.close()
    ready = all(check.is_ready for check in checks if check.required)
    response = ReadinessResponse(
        status="ready" if ready else "not_ready",
        is_ready=ready,
        app_name=settings.app_name,
        environment=settings.environment,
        retrieval_mode="hybrid",
        qdrant_required_for_current_mode=True,
        lexical_artifacts_required_for_current_mode=knowledge_mode == "combined",
        checks=checks,
        knowledge_mode=knowledge_mode,
    )
    if not ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=response.model_dump())
    return response


def _check_model_configuration(settings) -> ReadinessCheck:
    missing_fields: list[str] = []
    if not str(getattr(settings, "openai_api_key", "")).strip():
        missing_fields.append("OPENAI_API_KEY")
    if not str(getattr(settings, "openai_embedding_model", "")).strip():
        missing_fields.append("OPENAI_EMBEDDING_MODEL")
    if not str(getattr(settings, "openai_chat_model", "")).strip():
        missing_fields.append("OPENAI_CHAT_MODEL")

    if missing_fields:
        return ReadinessCheck(
            name="model_configuration",
            required=True,
            is_ready=False,
            detail="Missing required model configuration: " + ", ".join(missing_fields),
        )

    return ReadinessCheck(
        name="model_configuration",
        required=True,
        is_ready=True,
        detail="Embedding and chat model configuration is present. Model APIs are not called by readiness.",
    )


def _check_retrieval_ready_artifacts(
    processed_dir: str | Path,
    required: bool,
) -> ReadinessCheck:
    directory = Path(processed_dir)
    artifact_files = sorted(directory.glob("*.retrieval_ready.json")) if directory.exists() else []

    if artifact_files:
        return ReadinessCheck(
            name="retrieval_ready_artifacts",
            required=required,
            is_ready=True,
            detail=f"Found {len(artifact_files)} retrieval-ready artifact(s).",
        )

    if required:
        return ReadinessCheck(
            name="retrieval_ready_artifacts",
            required=True,
            is_ready=False,
            detail="No retrieval-ready artifacts found. Run ingestion before lexical or hybrid retrieval.",
        )

    return ReadinessCheck(
        name="retrieval_ready_artifacts",
        required=False,
        is_ready=True,
        detail="Retrieval-ready artifacts are not required for dense-only retrieval.",
    )


def _build_response(
    app_name: str,
    environment: str,
    retrieval_mode: str,
    qdrant_required: bool,
    lexical_required: bool,
    checks: list[ReadinessCheck],
) -> ReadinessResponse:
    is_ready = all(check.is_ready or not check.required for check in checks)
    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        is_ready=is_ready,
        app_name=app_name,
        environment=environment,
        retrieval_mode=retrieval_mode,
        qdrant_required_for_current_mode=qdrant_required,
        lexical_artifacts_required_for_current_mode=lexical_required,
        checks=checks,
    )


def _requires_qdrant_collection(retrieval_mode: str) -> bool:
    """Return whether the selected retrieval mode requires a Qdrant collection."""

    return retrieval_mode in {"dense", "hybrid"}


def _requires_lexical_artifacts(retrieval_mode: str) -> bool:
    """Return whether the selected retrieval mode requires retrieval-ready artifacts."""

    return retrieval_mode in {"lexical", "hybrid"}
