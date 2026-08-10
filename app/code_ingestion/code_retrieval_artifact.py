from __future__ import annotations

import hashlib

from antlr4 import CommonTokenStream, InputStream, Token

from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.plsql_models import (
    CodeRetrievalArtifact,
    CodeRetrievalUnit,
    ExtractedCodeNode,
    PackageContextSummary,
    PlSqlFileParseArtifact,
)
from app.code_ingestion.plsql_segmentation import build_fallback_segments


DERIVED_CONTEXT_MARKER = "DERIVED RETRIEVAL CONTEXT - NOT A CITATION SOURCE"
MAX_REFERENCES_PER_KIND = 20
MAX_DERIVED_CONTEXT_CHARACTERS = 2_000
_DECLARATION_KINDS = {"type", "constant", "global_variable", "cursor"}
_ROUTINE_KINDS = {"procedure", "procedure_spec", "function", "function_spec"}


def build_code_retrieval_artifact(
    parse_artifact: PlSqlFileParseArtifact,
    source_text: str,
    *,
    verified_source_sha256: str | None = None,
) -> CodeRetrievalArtifact:
    """Build citeable source units with small, explicitly derived context headers."""

    observed_hash = verified_source_sha256 or hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if observed_hash != parse_artifact.source_sha256:
        raise ValueError("source_text does not match parse artifact source_sha256")

    units_by_node_id: dict[str, CodeRetrievalUnit] = {}
    declarations = tuple(
        node for node in parse_artifact.extracted_nodes if node.node_kind in _DECLARATION_KINDS
    )
    declaration_unit_ids = {
        node.node_id: _unit_id(parse_artifact.snapshot_id, parse_artifact.source_path, node.node_id)
        for node in declarations
    }

    for node in parse_artifact.extracted_nodes:
        if node.node_kind in {"package", "package_body"}:
            continue
        original_text = _source_slice(source_text, node)
        derived_context = None
        related_unit_ids: tuple[str, ...] = ()
        retrieval_text = original_text
        if node.node_kind in _ROUTINE_KINDS:
            referenced = _referenced_declarations(original_text, declarations)
            related_unit_ids = tuple(declaration_unit_ids[item.node_id] for item in referenced)
            derived_context = _build_context(node, referenced)
            retrieval_text = _retrieval_text(derived_context, original_text)
        unit = CodeRetrievalUnit(
            unit_id=_unit_id(parse_artifact.snapshot_id, parse_artifact.source_path, node.node_id),
            source_kind=node.node_kind,
            snapshot_id=parse_artifact.snapshot_id,
            source_path=parse_artifact.source_path,
            source_map=node.source_map,
            display_name=node.display_name,
            package_name=node.package_name,
            text=original_text,
            retrieval_text=retrieval_text,
            derived_context=derived_context,
            related_unit_ids=related_unit_ids,
            parser_state=parse_artifact.parser_state,
            conditional_state=node.conditional_state,
        )
        units_by_node_id[node.node_id] = unit

    if not units_by_node_id:
        source_segments = tuple(
            segment
            for segment in parse_artifact.segments
            if segment.segment_kind == "fallback_chunk"
        ) or build_fallback_segments(source_text, source_path=parse_artifact.source_path)
        for segment in source_segments:
            text = source_text[segment.source_map.start_offset : segment.source_map.end_offset]
            source_kind = (
                "fallback_chunk"
                if parse_artifact.parser_state == "fallback_parse"
                else "source_chunk"
            )
            units_by_node_id[segment.segment_id] = CodeRetrievalUnit(
                unit_id=_unit_id(
                    parse_artifact.snapshot_id,
                    parse_artifact.source_path,
                    segment.segment_id,
                ),
                source_kind=source_kind,
                snapshot_id=parse_artifact.snapshot_id,
                source_path=parse_artifact.source_path,
                source_map=segment.source_map,
                display_name=segment.display_name or "fallback_chunk",
                text=text,
                retrieval_text=text,
                parser_state=parse_artifact.parser_state,
                conditional_state=(
                    "unresolved"
                    if parse_artifact.parser_state == "fallback_parse"
                    else "unconditional"
                ),
            )

    ordered = tuple(
        sorted(
            units_by_node_id.values(),
            key=lambda unit: (unit.source_map.start_offset, unit.source_kind, unit.unit_id),
        )
    )
    return CodeRetrievalArtifact(
        snapshot_id=parse_artifact.snapshot_id,
        source_path=parse_artifact.source_path,
        total_units=len(ordered),
        units=ordered,
    )


def _source_slice(source_text: str, node: ExtractedCodeNode) -> str:
    return source_text[node.source_map.start_offset : node.source_map.end_offset]


def _referenced_declarations(
    routine_text: str,
    declarations: tuple[ExtractedCodeNode, ...],
) -> tuple[ExtractedCodeNode, ...]:
    tokens = _canonical_source_tokens(routine_text)
    matches = [node for node in declarations if _canonical_identifier(node.display_name) in tokens]
    return tuple(
        sorted(
            matches,
            key=lambda node: (node.node_kind, node.source_map.start_offset, node.node_id),
        )
    )


def _canonical_source_tokens(source_text: str) -> set[str]:
    lexer = PlSqlLexer(InputStream(source_text))
    lexer.removeErrorListeners()
    stream = CommonTokenStream(lexer)
    stream.fill()
    return {
        _canonical_identifier(token.text)
        for token in stream.tokens
        if token.type != Token.EOF and token.channel == Token.DEFAULT_CHANNEL and token.text
    }


def _canonical_identifier(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value
    return value.upper()


def _build_context(
    routine: ExtractedCodeNode,
    declarations: tuple[ExtractedCodeNode, ...],
) -> PackageContextSummary:
    grouped: dict[str, list[str]] = {kind: [] for kind in _DECLARATION_KINDS}
    for declaration in declarations:
        names = grouped[declaration.node_kind]
        if declaration.display_name not in names and len(names) < MAX_REFERENCES_PER_KIND:
            names.append(declaration.display_name)
    return PackageContextSummary(
        package_name=routine.package_name or "<standalone>",
        public_signature=routine.signature_text,
        referenced_types=tuple(grouped["type"]),
        referenced_constants=tuple(grouped["constant"]),
        referenced_globals=tuple(grouped["global_variable"]),
        referenced_cursors=tuple(grouped["cursor"]),
        conditional_state=routine.conditional_state,
    )


def _retrieval_text(context: PackageContextSummary, original_text: str) -> str:
    lines = [
        DERIVED_CONTEXT_MARKER,
        f"Package: {context.package_name}",
        f"Public signature: {context.public_signature or '<unavailable>'}",
        f"Referenced package types: {_join(context.referenced_types)}",
        f"Referenced constants: {_join(context.referenced_constants)}",
        f"Referenced global variables: {_join(context.referenced_globals)}",
        f"Referenced cursors: {_join(context.referenced_cursors)}",
        f"Conditional compilation state: {context.conditional_state}",
    ]
    header = "\n".join(lines)
    if len(header) > MAX_DERIVED_CONTEXT_CHARACTERS:
        header = header[: MAX_DERIVED_CONTEXT_CHARACTERS - 3] + "..."
    return f"{header}\n\nORIGINAL CITATION SOURCE:\n{original_text}"


def _join(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "<none>"


def _unit_id(snapshot_id: str, source_path: str, local_id: str) -> str:
    return hashlib.sha256(
        f"code-unit-v1|{snapshot_id}|{source_path}|{local_id}".encode("utf-8")
    ).hexdigest()
