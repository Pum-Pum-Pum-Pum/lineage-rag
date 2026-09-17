"""Namespace-isolated, crash-safe records for coordinated update runs."""
from __future__ import annotations

from pathlib import Path

from app.code_updates.storage import Run as _Run, now, run_directory as _legacy_run_directory


def run_directory(root, run_id):
    """Validate IDs with the legacy rule, but never read legacy state."""
    legacy = _legacy_run_directory(root, run_id)
    return legacy.parents[1] / "knowledge_updates" / run_id


class Run(_Run):
    def __init__(self, root, run_id):
        # Do not call ``_Run.__init__``: it would read data/code_updates/<id>
        # before we substitute our directory, which can cross-contaminate two
        # independently resumable workflows with the same human release label.
        self.root = Path(root).resolve()
        self.directory = run_directory(self.root, run_id)
        self.path = self.directory / "state.json"
        from app.code_updates.storage import read
        self.state = read(self.path) if self.path.exists() else {
            "schema_version": "knowledge_update_run_v1", "run_id": run_id,
            "created_at": now(), "status": "NEW", "steps": {},
            "processing_seconds": 0.0,
        }
