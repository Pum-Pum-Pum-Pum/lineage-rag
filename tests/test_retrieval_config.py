from dataclasses import dataclass

from app.core.config import Settings
from app.retrieval.retrieval_config import (
    build_retrieval_runtime_config,
    is_hybrid_enabled,
    validate_retrieval_runtime_config,
)


@dataclass(frozen=True)
class _FakeSettings:
    retrieval_mode: str = "HYBRID"
    hybrid_dense_weight: float = 0.6
    hybrid_lexical_weight: float = 0.4
    hybrid_candidate_limit: int = 10


def test_settings_defaults_use_provisional_hybrid_configuration() -> None:
    settings = Settings(_env_file=None)

    assert settings.retrieval_mode == "hybrid"
    assert settings.hybrid_dense_weight == 0.60
    assert settings.hybrid_lexical_weight == 0.40
    assert settings.hybrid_candidate_limit == 10


def test_build_retrieval_runtime_config_normalizes_mode() -> None:
    config = build_retrieval_runtime_config(_FakeSettings())

    assert config.retrieval_mode == "hybrid"
    assert config.hybrid_dense_weight == 0.6
    assert config.hybrid_lexical_weight == 0.4
    assert config.hybrid_candidate_limit == 10
    assert is_hybrid_enabled(config) is True


def test_validate_retrieval_runtime_config_rejects_invalid_mode() -> None:
    try:
        validate_retrieval_runtime_config(
            retrieval_mode="agentic",
            hybrid_dense_weight=0.6,
            hybrid_lexical_weight=0.4,
            hybrid_candidate_limit=10,
        )
    except ValueError as exc:
        assert "Unsupported retrieval mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported retrieval mode")


def test_validate_retrieval_runtime_config_rejects_invalid_weights() -> None:
    try:
        validate_retrieval_runtime_config(
            retrieval_mode="hybrid",
            hybrid_dense_weight=0.0,
            hybrid_lexical_weight=0.0,
            hybrid_candidate_limit=10,
        )
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected ValueError for zero hybrid weights")


def test_validate_retrieval_runtime_config_rejects_invalid_candidate_limit() -> None:
    try:
        validate_retrieval_runtime_config(
            retrieval_mode="hybrid",
            hybrid_dense_weight=0.6,
            hybrid_lexical_weight=0.4,
            hybrid_candidate_limit=0,
        )
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid candidate limit")