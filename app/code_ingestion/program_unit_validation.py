from __future__ import annotations

from pathlib import Path

from app.code_ingestion.oracle_identifiers import oracle_identifier, qualified_parts
from app.code_ingestion.plsql_models import PlSqlFileParseArtifact


class ProgramUnitPolicyError(ValueError):
    """Raised when an uploaded PL/SQL file violates the custom-source contract."""


def validate_custom_program_unit(
    parse_artifact: PlSqlFileParseArtifact,
    *,
    source_handler: str,
    allowed_suffixes: tuple[str, ...],
) -> str | None:
    """Return the canonical top-level owner, or ``None`` for DDL sources.

    Package members inherit their package owner. Standalone procedures and
    functions are validated directly. The declaration is authoritative and
    the filename stem is a required intake assertion.
    """
    if source_handler == "ddl":
        return None
    owners: set[str] = set()
    for node in parse_artifact.extracted_nodes:
        if node.package_name:
            owners.add(_canonical_leaf(node.package_name))
            continue
        if node.node_kind in {
            "package",
            "package_body",
            "procedure",
            "procedure_spec",
            "function",
            "function_spec",
        } and not node.enclosing_routines:
            owners.add(_canonical_leaf(node.display_name))
    if not owners:
        raise ProgramUnitPolicyError(
            f"No top-level package, procedure, or function was extracted from {parse_artifact.source_path}"
        )
    if len(owners) != 1:
        raise ProgramUnitPolicyError(
            f"Expected exactly one top-level program unit in {parse_artifact.source_path}; found {sorted(owners)}"
        )
    owner = next(iter(owners))
    if not any(owner.endswith(suffix) for suffix in allowed_suffixes):
        raise ProgramUnitPolicyError(
            f"Top-level program unit {owner!r} must end with one of {allowed_suffixes}"
        )
    filename_stem = oracle_identifier(Path(parse_artifact.source_path).stem).canonical_name
    if filename_stem != owner:
        raise ProgramUnitPolicyError(
            f"Filename stem {filename_stem!r} does not match declared program unit {owner!r}"
        )
    return owner


def _canonical_leaf(value: str) -> str:
    return oracle_identifier(qualified_parts(value)[-1].display_name).canonical_name
