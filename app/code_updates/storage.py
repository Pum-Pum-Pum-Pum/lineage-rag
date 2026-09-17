"""Crash-safe local records. Locks are OS locks, not stale PID files."""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import portalocker


def now():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    atomic_text(Path(path), json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".update-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing immutable record differs: {path}")
        return
    write(path, value)


def run_directory(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("RunId must contain only letters, numbers, underscores or hyphens")
    path = Path(root).resolve() / "data/code_updates" / run_id
    if not path.resolve().is_relative_to(Path(root).resolve() / "data/code_updates"):
        raise ValueError("Run directory must stay inside data/code_updates")
    return path


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(str(directory / "run.lock"), timeout=0):
            yield
    except portalocker.exceptions.LockException as exc:
        raise RuntimeError("Another coordinator owns this run; wait for it to finish") from exc


class Run:
    def __init__(self, root, run_id):
        self.root = Path(root).resolve()
        self.directory = run_directory(root, run_id)
        self.path = self.directory / "state.json"
        self.state = read(self.path) if self.path.exists() else {
            "schema_version": "code_update_run_v1", "run_id": run_id,
            "created_at": now(), "status": "NEW", "steps": {}, "processing_seconds": 0.0,
        }

    def save(self):
        write(self.path, self.state)

    def event(self, kind, **details):
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), "event": kind, **details}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def transition(self, status, next_action):
        self.state.update(status=status, next_action=next_action)
        self.save()
        self.event("transition", status=status)

    def check_bindings(self):
        for path, expected in self.state.get("bindings", {}).items():
            if not Path(path).is_file() or sha(path) != expected:
                raise ValueError(f"Bound input changed: {path}. Create a new run; approvals cannot be reused.")
        for directory, binding in self.state.get("trees", {}).items():
            observed = sorted(str(p.relative_to(directory)) for p in Path(directory).rglob(binding["pattern"]) if p.is_file())
            # Directory enumeration order is not stable on Windows. Individual files are
            # separately hash-bound; this check is strictly for additions/removals.
            if len(observed) != len(binding["members"]) or set(observed) != set(binding["members"]):
                raise ValueError(f"Bound directory membership changed: {directory}. Create a new run.")

    def bind_tree(self, directory, pattern="*"):
        directory = Path(directory).resolve()
        files = sorted((p for p in directory.rglob(pattern) if p.is_file()), key=lambda p: str(p.relative_to(directory)))
        binding = {"pattern": pattern, "members": [str(p.relative_to(directory)) for p in files]}
        trees = self.state.setdefault("trees", {})
        previous = trees.get(str(directory))
        if previous is not None:
            if (previous["pattern"] != pattern or len(previous["members"]) != len(binding["members"])
                    or set(previous["members"]) != set(binding["members"])):
                raise ValueError(f"Bound directory membership changed: {directory}")
            binding = previous
        trees[str(directory)] = binding
        self.bind(files)

    def bind(self, paths):
        bindings = self.state.setdefault("bindings", {})
        for path in paths:
            path = str(Path(path).resolve())
            observed = sha(path)
            if path in bindings and bindings[path] != observed:
                raise ValueError(f"Bound input changed: {path}")
            bindings[path] = observed
        self.save()

    def step(self, name, operation):
        previous = self.state["steps"].get(name)
        if previous:
            for path, expected in previous["outputs"].items():
                if not Path(path).is_file() or sha(path) != expected:
                    raise ValueError(f"Checkpoint {name} has changed output: {path}")
            return previous["result"]
        self.event("step_started", step=name)
        result, outputs = operation()
        self.state["steps"][name] = {"result": result, "outputs": {
            str(Path(p).resolve()): sha(p) for p in outputs}, "completed_at": now()}
        self.save()
        self.event("step_completed", step=name)
        return result

    def summary(self):
        state = self.state
        action = state.get("next_action", "init")
        command = f".\\scripts\\run_code_update.ps1 -Action {action} -RunId {state['run_id']}"
        text = f"# Code update {state['run_id']}\n\nStatus: **{state['status']}**\n\nNext: `{command}`\n\n"
        text += "## Recorded execution\n\n"
        for key in ("created_at", "snapshot_id", "base_snapshot_id", "revision", "processing_seconds",
                    "waiting_seconds", "prompt_wait_seconds", "operator_prompts", "last_error", "final_collection", "promotion_hash"):
            if key in state:
                text += f"- {key}: {state[key]}\n"
        for name, entry in state["steps"].items():
            text += f"\n## {name}\n\nCompleted: {entry['completed_at']}\n\n"
            if entry.get("result"):
                text += "```json\n" + json.dumps(entry["result"], indent=2, ensure_ascii=False) + "\n```\n\n"
            for path in entry["outputs"]:
                relative = os.path.relpath(path, self.directory).replace("\\", "/")
                text += f"- [{Path(path).name}]({relative})\n"
        for name in ("config.json", "embedding_request.json", "review.md", "review.html",
                     "events.jsonl", "deferred_lineage.json"):
            if (self.directory / name).exists():
                text += f"\n[{name}]({name})\n"
        final = state.get("final", {})
        if final:
            decisions_path = Path(final["code"]).parent / "decisions.json"
            deferred_path = Path(final["code"]).parent / "deferred_lineage.json"
            if decisions_path.exists():
                choices = read(decisions_path)["decisions"].values()
                counts = {}
                for choice in choices:
                    verdict = "carried" if choice.get("prior_reviewer") else choice["verdict"]
                    counts[verdict] = counts.get(verdict, 0) + 1
                text += f"\nReview decisions: {json.dumps(counts)}\n"
            if deferred_path.exists():
                text += f"\nDeferred candidate relationships: {len(read(deferred_path))}. These are not reviewed implementation links.\n"
        atomic_text(self.directory / "summary.md", text)
        links = "".join(f'<li><a href="{html.escape(os.path.relpath(p, self.directory).replace(chr(92), chr(47)), quote=True)}">{html.escape(Path(p).name)}</a></li>'
                        for step in state["steps"].values() for p in step["outputs"])
        atomic_text(self.directory / "summary.html", '<!doctype html><meta charset="utf-8"><title>Code update</title>'
                    + '<pre style="white-space:pre-wrap">' + html.escape(text) + '</pre><ul>' + links + '</ul>')
        return command
