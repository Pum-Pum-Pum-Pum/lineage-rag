"""Additive FDD source planning with deterministic change classification."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from app.ingestion.filename_parser import parse_document_filename
from app.ingestion.fdd_document_lineage import load_fdd_document_lineage_policy
from app.knowledge_updates.models import ChangeSet, FddUpdatePlan, SourceIdentity, make_fdd_plan


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, relative_to: Path) -> SourceIdentity:
    parsed = parse_document_filename(path.name)
    return SourceIdentity(path=path.relative_to(relative_to).as_posix(), sha256=_sha(path),
        size_bytes=path.stat().st_size, logical_key=parsed.document_lineage_key,
        revision=parsed.document_revision)


def _files(directory: Path) -> tuple[SourceIdentity, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(f"FDD source directory does not exist: {directory}")
    root = directory.resolve()
    files = []
    for path in directory.rglob("*.docx"):
        # A source update is copied into controlled storage.  A symlink or
        # junction can change what is read after planning, or escape the
        # operator-provided tree, so it is never accepted as evidence.
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"FDD source symlink/junction escape is not permitted: {path}")
        files.append(path)
    identities = tuple(sorted((_identity(path, relative_to=directory) for path in files), key=lambda i: i.path.casefold()))
    names = [item.path.casefold() for item in identities]
    if len(names) != len(set(names)):
        raise ValueError("FDD source paths conflict case-insensitively")
    return identities


def _current(identities: Iterable[SourceIdentity], policy) -> tuple[SourceIdentity, ...]:
    # The production selector operates on files. Replicate only its documented
    # latest-by-lineage behavior here so planning stays read-only.
    grouped: dict[str, list[SourceIdentity]] = {}
    independent: list[SourceIdentity] = []
    for item in identities:
        if item.revision is None or item.logical_key.casefold() not in policy.full_replacement_keys:
            independent.append(item)
        else:
            grouped.setdefault(item.logical_key.casefold(), []).append(item)
    selected = independent[:]
    for values in grouped.values():
        selected.append(max(values, key=lambda i: tuple(int(v) for v in (i.revision or "0").split("."))))
    return tuple(sorted(selected, key=lambda i: i.path.casefold()))


def plan_additive_fdd_update(*, base_generation: str, target_generation: str,
                             baseline_directory: Path, update_directory: Path,
                             withdrawn: Iterable[str] = (), replacements: Iterable[str] = ()) -> FddUpdatePlan:
    """Plan an additive update without copying, embedding, or changing a store."""
    if base_generation == target_generation:
        raise ValueError("Target FDD generation must be new")
    policy = load_fdd_document_lineage_policy()
    baseline = _files(baseline_directory)
    updates = _files(update_directory)
    withdrawn_folded = {item.replace("\\", "/").casefold() for item in withdrawn}
    replacement_folded = {item.replace("\\", "/").casefold() for item in replacements}
    baseline_by_path = {item.path.casefold(): item for item in baseline}
    update_by_path = {item.path.casefold(): item for item in updates}
    duplicate_conflicts = [name for name in baseline_by_path.keys() & update_by_path.keys()
                           if baseline_by_path[name].sha256 != update_by_path[name].sha256]
    # Same source name with different bytes is only legal if it is a configured
    # logical replacement. Operators must otherwise give the document a new ID.
    for name in duplicate_conflicts:
        old, new = baseline_by_path[name], update_by_path[name]
        if (old.logical_key.casefold() != new.logical_key.casefold() or new.revision is None
                or name not in replacement_folded):
            raise ValueError(f"Conflicting FDD source identity: {new.path}")
    combined = list(baseline)
    for item in updates:
        old = baseline_by_path.get(item.path.casefold())
        if old is not None:
            combined.remove(old)
        combined.append(item)
    selected = tuple(item for item in _current(combined, policy) if item.path.casefold() not in withdrawn_folded)
    before_by_hash = {item.sha256: item for item in baseline}
    target_paths = {item.path.casefold() for item in selected}
    added, modified, unchanged, moved = [], [], [], []
    for item in selected:
        old = baseline_by_path.get(item.path.casefold())
        if old and old.sha256 == item.sha256:
            unchanged.append(item.path)
        elif old:
            modified.append(item.path)
        elif item.sha256 in before_by_hash:
            moved.append((before_by_hash[item.sha256].path, item.path))
        else:
            added.append(item.path)
    # A full-replacement document is *superseded*, not additionally removed.
    # Change classes are deliberately disjoint because a reviewer needs one
    # clear action for each prior source.
    superseded_folded = {
        item.path.casefold() for item in combined
        if item.path.casefold() not in target_paths and item.path.casefold() not in withdrawn_folded
    }
    removed = [item.path for item in baseline if item.path.casefold() not in target_paths
               and item.path.casefold() not in withdrawn_folded
               and item.path.casefold() not in superseded_folded]
    superseded = [item.path for item in combined if item.path.casefold() in superseded_folded]
    return make_fdd_plan(base_generation=base_generation, target_generation=target_generation,
        source_policy_sha256=policy.sha256, baseline_sources=baseline, update_sources=updates,
        target_sources=selected, changes=ChangeSet(added=tuple(sorted(added)), modified=tuple(sorted(modified)),
        unchanged=tuple(sorted(unchanged)), moved=tuple(sorted(moved)), removed=tuple(sorted(removed)),
        superseded=tuple(sorted(superseded)), withdrawn=tuple(sorted(withdrawn_folded))))
