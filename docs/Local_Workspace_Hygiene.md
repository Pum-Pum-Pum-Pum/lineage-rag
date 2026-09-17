# Local workspace and Git

Generated development output belongs under the ignored `.local/` directory:

```text
.local/
  pytest/
    cache/       pytest's reusable cache
    runs/        one unique temporary fixture directory per test invocation
    archive/     existing root-level .pytest* folders, retained as evidence
  codex/
    archive/     existing ._codex* scratch/test folders, retained as evidence
  uv/
    archive/     historical local uv cache folders
```

Run tests from the repository root as usual:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

`pytest.ini` places the cache here; `tests/conftest.py` assigns a unique fixture
directory. These settings affect tests only. An explicit `--basetemp` overrides
the default; always use a new disposable directory because pytest clears it.
Keep future assistant scratch work in `.local/codex/`, not in the project root.
The real `.codex/` and `.agents/` configuration locations must not be relocated.

Archived test outputs can contain obsolete absolute paths. They are historical
diagnostics, not resumable ingestion runs. Runtime snapshots, update runs, source
archives, vector stores, `.env`, and the virtual environment stay in their original
locations. In particular, do not move `data/code_updates/` or `data/knowledge_updates/`.

Git ignores local development output, update-run artifacts, credentials and vector
stores. Source code, tests, docs, configuration templates and reviewed evaluation
fixtures remain eligible for version control. Ignore rules do not remove files
already tracked: the cleanup stages removal of legacy generated test output and
the local code-vector-store metadata from Git's index, keeping local data.
Previously committed files remain in Git history.

Before committing, inspect the staged changes and the dry-run file list:

```powershell
git diff --cached --stat
git add --dry-run .
```

LF/CRLF messages are line-ending warnings; inaccessible generated folders and
`Filename too long` were the Git-add blockers. Keep generated outputs ignored
instead of adding them to the repository. No Git history rewrite is needed.
