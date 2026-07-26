from pathlib import Path


def test_readme_documents_api_run_and_smoke_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "uv run --locked uvicorn app.api.main:app --reload" in readme
    assert "uv run --locked python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000" in readme
    assert "--check-ready" in readme
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
    assert "uv run --locked python scripts/run_qdrant_indexing.py" in readme
    assert "insufficient-evidence response" in readme
    assert "avoids printing raw server response bodies" in readme
    assert "secrets, stack traces, local file paths, or internal configuration values" in readme


def test_readme_documents_readiness_endpoint_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "uv run --locked python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --check-ready" in readme
    assert "dependency/artifact readiness check" in readme
    assert "checks model configuration is present without calling model APIs" in readme
    assert "checks `.retrieval_ready.json` artifacts when lexical or hybrid retrieval needs local lexical evidence" in readme
    assert "checks Qdrant collection existence when dense or hybrid retrieval needs vector search" in readme
    assert "does **not** run retrieval, embeddings, LLM generation" in readme
    assert "when `--check-ready` fails, the smoke-test client stops before any optional `POST /query`" in readme
    assert "stops before `/query` when `/ready` fails" in readme
    assert "Missing corpus evidence during a real query should produce an insufficient-evidence/refusal response" in readme


def test_readme_documents_retrieval_mode_dependency_matrix() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "### Retrieval-mode dependency matrix" in readme
    assert "| `lexical` |" in readme
    assert "| `dense` |" in readme
    assert "| `hybrid` |" in readme
    assert "local lexical artifacts only; no Qdrant client or collection check" in readme
    assert "no embedding call for retrieval" in readme
    assert "Qdrant collection and embedding call for dense vector search" in readme
    assert "both Qdrant dense search and local lexical artifacts" in readme
    assert "must fail fast when vector-store state is unavailable" in readme
    assert "avoids wasted model spend and misleading answers" in readme
