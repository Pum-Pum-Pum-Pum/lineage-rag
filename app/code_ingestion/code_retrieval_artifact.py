from __future__ import annotations

import bisect
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
    max_unit_characters: int = 6_000,
    overlap_characters: int = 400,
) -> CodeRetrievalArtifact:
    """Build citeable source units with small, explicitly derived context headers."""

    observed_hash = verified_source_sha256 or hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if observed_hash != parse_artifact.source_sha256:
        raise ValueError("source_text does not match parse artifact source_sha256")
    if max_unit_characters <= 0 or overlap_characters < 0:
        raise ValueError("Retrieval chunk bounds are invalid")
    if overlap_characters >= max_unit_characters:
        raise ValueError("overlap_characters must be smaller than max_unit_characters")

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
        parent_unit_id = _unit_id(
            parse_artifact.snapshot_id, parse_artifact.source_path, node.node_id
        )
        unit_values = dict(
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
        retrieval_overhead = (
            len(_retrieval_text(derived_context, "")) if derived_context is not None else 0
        )
        source_character_bound = max_unit_characters - retrieval_overhead
        if source_character_bound <= 0:
            raise ValueError("Derived retrieval context leaves no room for source text")
        if len(retrieval_text) <= max_unit_characters:
            units_by_node_id[node.node_id] = CodeRetrievalUnit(
                unit_id=parent_unit_id,
                **unit_values,
            )
            continue
        child_ranges = _bounded_ranges(
            source_text,
            node.source_map.start_offset,
            node.source_map.end_offset,
            max_characters=source_character_bound,
            overlap_characters=overlap_characters,
        )
        for chunk_index, (start_offset, end_offset) in enumerate(child_ranges):
            child_text = source_text[start_offset:end_offset]
            child_map = _source_map_for_offsets(
                source_text,
                parse_artifact.source_path,
                start_offset,
                end_offset,
            )
            child_key = f"{node.node_id}:chunk:{chunk_index}"
            child_context = derived_context
            child_retrieval_text = (
                _retrieval_text(child_context, child_text)
                if child_context is not None
                else child_text
            )
            units_by_node_id[child_key] = CodeRetrievalUnit(
                unit_id=_child_unit_id(parent_unit_id, chunk_index, start_offset, end_offset),
                parent_unit_id=parent_unit_id,
                parent_source_map=node.source_map,
                chunk_index=chunk_index,
                chunk_count=len(child_ranges),
                **{
                    **unit_values,
                    "source_map": child_map,
                    "text": child_text,
                    "retrieval_text": child_retrieval_text,
                },
            )

    covered_ranges = {
        (
            (unit.parent_source_map or unit.source_map).start_offset,
            (unit.parent_source_map or unit.source_map).end_offset,
        )
        for unit in units_by_node_id.values()
    }
    for segment in parse_artifact.segments:
        source_range = (segment.source_map.start_offset, segment.source_map.end_offset)
        if segment.parse_succeeded or source_range in covered_ranges:
            continue
        if segment.segment_kind == "fallback_chunk":
            continue
        _add_bounded_segment_units(
            units_by_node_id,
            parse_artifact=parse_artifact,
            source_text=source_text,
            local_id=segment.segment_id,
            source_map=segment.source_map,
            source_kind="degraded_routine",
            display_name=segment.display_name or "degraded_routine",
            max_unit_characters=max_unit_characters,
            overlap_characters=overlap_characters,
        )

    if not units_by_node_id:
        source_segments = tuple(
            segment
            for segment in parse_artifact.segments
            if segment.segment_kind == "fallback_chunk"
        ) or build_fallback_segments(source_text, source_path=parse_artifact.source_path)
        for segment in source_segments:
            source_kind = (
                "fallback_chunk"
                if parse_artifact.parser_state == "fallback_parse"
                else "source_chunk"
            )
            _add_bounded_segment_units(
                units_by_node_id,
                parse_artifact=parse_artifact,
                source_text=source_text,
                local_id=segment.segment_id,
                source_map=segment.source_map,
                source_kind=source_kind,
                display_name=segment.display_name or "fallback_chunk",
                max_unit_characters=max_unit_characters,
                overlap_characters=overlap_characters,
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
        max_unit_characters=max_unit_characters,
        overlap_characters=overlap_characters,
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


def _child_unit_id(parent_unit_id: str, index: int, start: int, end: int) -> str:
    return hashlib.sha256(
        f"code-unit-child-v1|{parent_unit_id}|{index}|{start}|{end}".encode("utf-8")
    ).hexdigest()


def _add_bounded_segment_units(
    target: dict[str, CodeRetrievalUnit],
    *,
    parse_artifact: PlSqlFileParseArtifact,
    source_text: str,
    local_id: str,
    source_map,
    source_kind: str,
    display_name: str,
    max_unit_characters: int,
    overlap_characters: int,
) -> None:
    parent_unit_id = _unit_id(
        parse_artifact.snapshot_id,
        parse_artifact.source_path,
        local_id,
    )
    ranges = _bounded_ranges(
        source_text,
        source_map.start_offset,
        source_map.end_offset,
        max_characters=max_unit_characters,
        overlap_characters=overlap_characters,
    )
    for index, (start, end) in enumerate(ranges):
        child_map = _source_map_for_offsets(
            source_text, parse_artifact.source_path, start, end
        )
        text = source_text[start:end]
        is_child = len(ranges) > 1
        key = local_id if not is_child else f"{local_id}:chunk:{index}"
        target[key] = CodeRetrievalUnit(
            unit_id=(
                parent_unit_id
                if not is_child
                else _child_unit_id(parent_unit_id, index, start, end)
            ),
            parent_unit_id=parent_unit_id if is_child else None,
            parent_source_map=source_map if is_child else None,
            chunk_index=index if is_child else None,
            chunk_count=len(ranges) if is_child else None,
            source_kind=source_kind,
            snapshot_id=parse_artifact.snapshot_id,
            source_path=parse_artifact.source_path,
            source_map=child_map,
            display_name=display_name,
            text=text,
            retrieval_text=text,
            parser_state=parse_artifact.parser_state,
            conditional_state=(
                "unresolved"
                if parse_artifact.parser_state == "fallback_parse"
                else "unconditional"
            ),
        )


def _bounded_ranges(
    source_text: str,
    start_offset: int,
    end_offset: int,
    *,
    max_characters: int,
    overlap_characters: int,
) -> tuple[tuple[int, int], ...]:
    """Create deterministic, gap-free ranges with bounded overlap."""

    ranges: list[tuple[int, int]] = []
    start = start_offset
    while start < end_offset:
        end = min(start + max_characters, end_offset)
        if end < end_offset:
            newline = source_text.rfind("\n", start + max_characters // 2, end)
            if newline >= start:
                end = newline + 1
        ranges.append((start, end))
        if end == end_offset:
            break
        next_start = max(start + 1, end - overlap_characters)
        newline = source_text.find("\n", next_start, end)
        if newline >= 0 and newline + 1 < end:
            next_start = newline + 1
        start = next_start
    return tuple(ranges)


def _source_map_for_offsets(
    source_text: str,
    source_path: str,
    start_offset: int,
    end_offset: int,
):
    line_starts = [0]
    line_starts.extend(index + 1 for index, character in enumerate(source_text) if character == "\n")
    start_line = bisect.bisect_right(line_starts, start_offset)
    last_character_offset = max(start_offset, end_offset - 1)
    end_line = bisect.bisect_right(line_starts, last_character_offset)
    from app.code_ingestion.plsql_models import SourceMap

    return SourceMap(
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        start_offset=start_offset,
        end_offset=end_offset,
    )
