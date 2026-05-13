from pathlib import Path


def test_readme_documents_api_run_and_smoke_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python -m uvicorn app.api.main:app --reload" in readme
    assert "python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000" in readme
    assert "GET /health" in readme
    assert "GET /ready" in readme
    assert "POST /query" in readme
    assert "run_grounded_answer_query" in readme
    assert "Dense and hybrid retrieval require a Qdrant collection" in readme
    assert "Lexical-only retrieval uses local `.retrieval_ready.json` artifacts" in readme


def test_readme_documents_smoke_test_operational_interpretation() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "No query supplied. Skipped POST /query." in readme
    assert "does **not** run retrieval, embeddings, LLM generation" in readme
    assert "may trigger retrieval, embedding calls, LLM generation" in readme
    assert "answer status, evidence sufficiency, refusal reason" in readme
    assert "data/exports/answer_runs/" in readme


def test_readme_documents_safe_failure_interpretation() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "### Smoke-test failure interpretation" in readme
    assert "If `/query` returns `503` in dense or hybrid mode" in readme
    assert "python scripts/run_qdrant_indexing.py" in readme
    assert "insufficient-evidence response" in readme
    assert "avoids printing raw server response bodies" in readme
    assert "secrets, stack traces, local file paths, or internal configuration values" in readme


def test_readme_documents_readiness_endpoint_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "curl http://127.0.0.1:8000/ready" in readme
    assert "dependency/artifact readiness check" in readme
    assert "checks model configuration is present without calling model APIs" in readme
    assert "checks `.retrieval_ready.json` artifacts when lexical or hybrid retrieval needs local lexical evidence" in readme
    assert "checks Qdrant collection existence when dense or hybrid retrieval needs vector search" in readme
    assert "does **not** run retrieval, embeddings, LLM generation" in readme
    assert "Missing corpus evidence during a real query should produce an insufficient-evidence/refusal response" in readme