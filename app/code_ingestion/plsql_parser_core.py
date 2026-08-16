from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from antlr4 import CommonTokenStream, InputStream, ParserRuleContext
from antlr4.error.ErrorListener import ErrorListener

from app.code_ingestion.conditional_compilation import (
    build_conditional_parse_view,
    conditional_state_for_range,
)
from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.generated.plsql.PlSqlParser import PlSqlParser
from app.code_ingestion.plsql_models import (
    ExtractedCodeNode,
    ParseDiagnostic,
    ParsedSegment,
    PlSqlFileParseArtifact,
    SourceMap,
)
from app.code_ingestion.plsql_segmentation import (
    build_fallback_segments,
    detect_package_name,
    find_routine_segments,
)
from app.code_ingestion.snapshot_models import CompilerContext


@dataclass(frozen=True)
class _ParseAttempt:
    tree: ParserRuleContext
    tokens: CommonTokenStream
    diagnostics: tuple[ParseDiagnostic, ...]
    syntax_error_count: int


class _CollectingErrorListener(ErrorListener):
    def __init__(self, stage: str, *, limit: int = 100) -> None:
        self.stage = stage
        self.limit = limit
        self.diagnostics: list[ParseDiagnostic] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):  # noqa: N802
        if len(self.diagnostics) >= self.limit:
            return
        self.diagnostics.append(
            ParseDiagnostic(
                stage=self.stage,  # type: ignore[arg-type]
                severity="error",
                code="antlr_syntax_error",
                message=str(msg)[:500],
                line=line,
                column=column,
            )
        )


def parse_plsql_source(
    source_text: str,
    *,
    snapshot_id: str,
    source_path: str,
    source_sha256: str,
    compiler_context: CompilerContext | None = None,
    max_segment_characters: int = 500,
) -> PlSqlFileParseArtifact:
    started = time.perf_counter()
    conditional_view = build_conditional_parse_view(
        source_text,
        source_path=source_path,
        compiler_context=compiler_context,
    )
    raw_attempt = _parse_sql_script(source_text, stage="full_parse")
    selected_attempt = raw_attempt
    parse_text = source_text
    diagnostics = list(conditional_view.diagnostics) + list(raw_attempt.diagnostics)

    if raw_attempt.syntax_error_count and conditional_view.regions:
        view_attempt = _parse_sql_script(conditional_view.text, stage="full_parse")
        if view_attempt.syntax_error_count < raw_attempt.syntax_error_count:
            selected_attempt = view_attempt
            parse_text = conditional_view.text
            diagnostics.append(
                ParseDiagnostic(
                    stage="full_parse",
                    severity="warning",
                    code="conditional_parse_view_used",
                    message=(
                        "The raw grammar parse failed around conditional compilation; "
                        "a line-preserving directive view was used while original source remained citeable."
                    ),
                )
            )

    if selected_attempt.syntax_error_count == 0:
        nodes = _extract_nodes(
            selected_attempt.tree,
            source_text=source_text,
            source_path=source_path,
            conditional_regions=conditional_view.regions,
        )
        file_segment = ParsedSegment(
            segment_id=_segment_id(source_path, 0, len(source_text), "file"),
            segment_kind="file",
            source_map=_whole_file_map(source_text, source_path),
            parse_succeeded=True,
            syntax_error_count=0,
        )
        return PlSqlFileParseArtifact(
            snapshot_id=snapshot_id,
            source_path=source_path,
            source_sha256=source_sha256,
            parser_state="full_parse",
            duration_ms=(time.perf_counter() - started) * 1000,
            peak_memory_bytes=0,
            syntax_error_count=raw_attempt.syntax_error_count,
            conditional_regions=conditional_view.regions,
            conditional_error_directives=conditional_view.error_directives,
            segments=(file_segment,),
            extracted_nodes=nodes,
            diagnostics=tuple(diagnostics),
        )

    segmented = _parse_segments(
        parse_text,
        original_source=source_text,
        source_path=source_path,
        conditional_regions=conditional_view.regions,
        max_segment_characters=max_segment_characters,
    )
    successful_segments = tuple(segment for segment, _ in segmented if segment.parse_succeeded)
    segmented_nodes = tuple(node for _, nodes in segmented for node in nodes)
    if segmented_nodes:
        skipped_count = sum(
            segment.degradation_reason == "segment_character_limit_exceeded"
            for segment, _ in segmented
        )
        diagnostics.append(
            ParseDiagnostic(
                stage="segmented_parse",
                severity="warning",
                code="full_parse_degraded_to_segments",
                message=(
                    "Full-file parsing failed; routine segments were retained through "
                    f"ANTLR or conservative token structure ({skipped_count} oversized)."
                ),
            )
        )
        return PlSqlFileParseArtifact(
            snapshot_id=snapshot_id,
            source_path=source_path,
            source_sha256=source_sha256,
            parser_state="segmented_parse",
            duration_ms=(time.perf_counter() - started) * 1000,
            peak_memory_bytes=0,
            syntax_error_count=selected_attempt.syntax_error_count,
            conditional_regions=conditional_view.regions,
            conditional_error_directives=conditional_view.error_directives,
            segments=tuple(segment for segment, _ in segmented),
            extracted_nodes=segmented_nodes,
            diagnostics=tuple(diagnostics),
        )

    fallback = build_fallback_segments(source_text, source_path=source_path)
    diagnostics.append(
        ParseDiagnostic(
            stage="fallback",
            severity="warning",
            code="structural_parser_fallback",
            message="No routine segment parsed successfully; bounded original-source line chunks were retained.",
        )
    )
    return PlSqlFileParseArtifact(
        snapshot_id=snapshot_id,
        source_path=source_path,
        source_sha256=source_sha256,
        parser_state="fallback_parse",
        duration_ms=(time.perf_counter() - started) * 1000,
        peak_memory_bytes=0,
        syntax_error_count=selected_attempt.syntax_error_count,
        conditional_regions=conditional_view.regions,
        conditional_error_directives=conditional_view.error_directives,
        segments=fallback,
        diagnostics=tuple(diagnostics),
    )


def parse_plsql_segments_only(
    source_text: str,
    *,
    snapshot_id: str,
    source_path: str,
    source_sha256: str,
    compiler_context: CompilerContext | None = None,
    max_segment_characters: int = 500,
) -> PlSqlFileParseArtifact:
    """Run the token-aware segmented path without attempting a full-file parse."""

    started = time.perf_counter()
    conditional_view = build_conditional_parse_view(
        source_text,
        source_path=source_path,
        compiler_context=compiler_context,
    )
    segmented = _parse_segments(
        conditional_view.text,
        original_source=source_text,
        source_path=source_path,
        conditional_regions=conditional_view.regions,
        max_segment_characters=max_segment_characters,
    )
    nodes = tuple(node for _, extracted in segmented for node in extracted)
    syntax_errors = sum(segment.syntax_error_count for segment, _ in segmented)
    diagnostics = list(conditional_view.diagnostics)
    if nodes:
        parsed_count = sum(segment.parse_succeeded for segment, _ in segmented)
        skipped_count = sum(
            segment.degradation_reason == "segment_character_limit_exceeded"
            for segment, _ in segmented
        )
        failed_count = len(segmented) - parsed_count - skipped_count
        diagnostics.append(
            ParseDiagnostic(
                stage="segmented_parse",
                severity="warning",
                code=(
                    "segmented_parse_partial_recovery"
                    if failed_count or skipped_count
                    else "segmented_parse_resource_recovery"
                ),
                message=(
                    f"Token-aware segmentation retained {parsed_count} ANTLR-parsed and "
                    f"{skipped_count} structurally recovered oversized routines; "
                    f"{failed_count} candidate routines did not parse."
                ),
            )
        )
        return PlSqlFileParseArtifact(
            snapshot_id=snapshot_id,
            source_path=source_path,
            source_sha256=source_sha256,
            parser_state="segmented_parse",
            duration_ms=(time.perf_counter() - started) * 1000,
            peak_memory_bytes=0,
            syntax_error_count=syntax_errors,
            conditional_regions=conditional_view.regions,
            conditional_error_directives=conditional_view.error_directives,
            segments=tuple(segment for segment, _ in segmented),
            extracted_nodes=nodes,
            diagnostics=tuple(diagnostics),
        )

    fallback = build_fallback_segments(source_text, source_path=source_path)
    diagnostics.append(
        ParseDiagnostic(
            stage="fallback",
            severity="warning",
            code="segmented_parser_fallback",
            message=(
                "The bounded segmented attempt retained no trustworthy routine; "
                "bounded original-source chunks were preserved."
            ),
        )
    )
    return PlSqlFileParseArtifact(
        snapshot_id=snapshot_id,
        source_path=source_path,
        source_sha256=source_sha256,
        parser_state="fallback_parse",
        duration_ms=(time.perf_counter() - started) * 1000,
        peak_memory_bytes=0,
        syntax_error_count=syntax_errors,
        conditional_regions=conditional_view.regions,
        conditional_error_directives=conditional_view.error_directives,
        segments=fallback,
        diagnostics=tuple(diagnostics),
    )


def _parse_sql_script(source_text: str, *, stage: str) -> _ParseAttempt:
    lexer = PlSqlLexer(InputStream(source_text))
    lexer_listener = _CollectingErrorListener(stage)
    lexer.removeErrorListeners()
    lexer.addErrorListener(lexer_listener)
    tokens = CommonTokenStream(lexer)
    parser = PlSqlParser(tokens)
    parser_listener = _CollectingErrorListener(stage)
    parser.removeErrorListeners()
    parser.addErrorListener(parser_listener)
    tree = parser.sql_script()
    diagnostics = tuple(lexer_listener.diagnostics + parser_listener.diagnostics)
    return _ParseAttempt(
        tree=tree,
        tokens=tokens,
        diagnostics=diagnostics,
        syntax_error_count=len(diagnostics),
    )


def _parse_segments(
    parse_text: str,
    *,
    original_source: str,
    source_path: str,
    conditional_regions,
    max_segment_characters: int,
) -> tuple[tuple[ParsedSegment, tuple[ExtractedCodeNode, ...]], ...]:
    candidates = find_routine_segments(parse_text, source_path=source_path)
    package_name = detect_package_name(parse_text)
    results: list[tuple[ParsedSegment, tuple[ExtractedCodeNode, ...]]] = []
    for candidate in candidates:
        fragment = parse_text[
            candidate.source_map.start_offset : candidate.source_map.end_offset
        ]
        if len(fragment) > max_segment_characters:
            updated = candidate.model_copy(
                update={"degradation_reason": "segment_character_limit_exceeded"}
            )
            structural_node = _structural_node_from_segment(
                candidate,
                original_source=original_source,
                source_path=source_path,
                package_name=package_name,
                conditional_regions=conditional_regions,
            )
            results.append((updated, (structural_node,)))
            continue
        lexer = PlSqlLexer(InputStream(fragment))
        listener = _CollectingErrorListener("segmented_parse")
        lexer.removeErrorListeners()
        lexer.addErrorListener(listener)
        tokens = CommonTokenStream(lexer)
        parser = PlSqlParser(tokens)
        parser.removeErrorListeners()
        parser.addErrorListener(listener)
        if candidate.segment_kind == "procedure":
            tree = parser.procedure_body()
        elif candidate.segment_kind == "function":
            tree = parser.function_body()
        elif candidate.segment_kind == "procedure_spec":
            tree = parser.procedure_spec()
        else:
            tree = parser.function_spec()
        succeeded = not listener.diagnostics
        updated = candidate.model_copy(
            update={
                "parse_succeeded": succeeded,
                "syntax_error_count": len(listener.diagnostics),
            }
        )
        nodes = ()
        if succeeded:
            nodes = _extract_nodes(
                tree,
                source_text=original_source,
                source_path=source_path,
                conditional_regions=conditional_regions,
                offset_adjustment=candidate.source_map.start_offset,
                line_adjustment=candidate.source_map.start_line - 1,
                package_name_override=package_name,
            )
        results.append((updated, nodes))
    return tuple(results)


def _structural_node_from_segment(
    segment: ParsedSegment,
    *,
    original_source: str,
    source_path: str,
    package_name: str | None,
    conditional_regions,
) -> ExtractedCodeNode:
    """Retain conservative lexer-proven routine identity when ANTLR is bounded out."""

    node_kind = segment.segment_kind
    if node_kind not in {"procedure", "procedure_spec", "function", "function_spec"}:
        raise ValueError(f"Unsupported structural routine kind: {node_kind}")
    return ExtractedCodeNode(
        node_id=_segment_id(
            source_path,
            segment.source_map.start_offset,
            segment.source_map.end_offset,
            f"token_structural_{node_kind}",
        ),
        node_kind=node_kind,
        display_name=segment.display_name or "<unresolved>",
        package_name=package_name,
        extraction_method="token_structural",
        source_map=segment.source_map,
        signature_text=_signature_text(
            original_source[
                segment.source_map.start_offset : segment.source_map.end_offset
            ],
            is_spec=node_kind.endswith("_spec"),
        ),
        conditional_state=conditional_state_for_range(
            conditional_regions,
            start_offset=segment.source_map.start_offset,
            end_offset=segment.source_map.end_offset,
        ),
    )


def _extract_nodes(
    tree: ParserRuleContext,
    *,
    source_text: str,
    source_path: str,
    conditional_regions,
    offset_adjustment: int = 0,
    line_adjustment: int = 0,
    package_name_override: str | None = None,
) -> tuple[ExtractedCodeNode, ...]:
    nodes: list[ExtractedCodeNode] = []

    def visit(
        context: ParserRuleContext,
        package_name: str | None,
        routine_depth: int,
        routine_stack: tuple[str, ...],
    ) -> None:
        current_package = package_name
        node_kind: str | None = None
        display_name: str | None = None
        signature: str | None = None
        is_routine = isinstance(
            context,
            (
                PlSqlParser.Create_procedure_bodyContext,
                PlSqlParser.Create_function_bodyContext,
                PlSqlParser.Procedure_bodyContext,
                PlSqlParser.Function_bodyContext,
                PlSqlParser.Procedure_specContext,
                PlSqlParser.Function_specContext,
            ),
        )
        if isinstance(context, (PlSqlParser.Create_packageContext, PlSqlParser.Create_package_bodyContext)):
            package_names = context.package_name()
            package_ctx = package_names[0] if isinstance(package_names, list) else package_names
            current_package = package_ctx.getText() if package_ctx is not None else package_name_override
            node_kind = "package_body" if isinstance(context, PlSqlParser.Create_package_bodyContext) else "package"
            display_name = current_package
        elif isinstance(context, PlSqlParser.Procedure_bodyContext):
            node_kind, display_name = "procedure", context.identifier().getText()
        elif isinstance(context, PlSqlParser.Function_bodyContext):
            node_kind, display_name = "function", context.identifier().getText()
        elif isinstance(context, PlSqlParser.Create_procedure_bodyContext):
            node_kind, display_name = "procedure", context.procedure_name().getText()
        elif isinstance(context, PlSqlParser.Create_function_bodyContext):
            node_kind, display_name = "function", context.function_name().getText()
        elif isinstance(context, PlSqlParser.Procedure_specContext):
            node_kind, display_name = "procedure_spec", context.identifier().getText()
        elif isinstance(context, PlSqlParser.Function_specContext):
            node_kind, display_name = "function_spec", context.identifier().getText()
        elif routine_depth == 0 and isinstance(context, PlSqlParser.Type_declarationContext):
            node_kind, display_name = "type", context.identifier().getText()
        elif routine_depth == 0 and isinstance(context, PlSqlParser.Cursor_declarationContext):
            node_kind, display_name = "cursor", context.identifier().getText()
        elif routine_depth == 0 and isinstance(context, PlSqlParser.Variable_declarationContext):
            node_kind = "constant" if context.CONSTANT() is not None else "global_variable"
            display_name = context.identifier().getText()

        if node_kind and display_name and context.start is not None and context.stop is not None:
            start_offset = context.start.start + offset_adjustment
            end_offset = context.stop.stop + 1 + offset_adjustment
            start_line = context.start.line + line_adjustment
            end_line = context.stop.line + line_adjustment
            if is_routine:
                signature = _signature_text(
                    source_text[start_offset:end_offset],
                    is_spec=node_kind.endswith("_spec"),
                )
            source_map = SourceMap(
                source_path=source_path,
                start_line=start_line,
                end_line=end_line,
                start_offset=start_offset,
                end_offset=end_offset,
            )
            nodes.append(
                ExtractedCodeNode(
                    node_id=_segment_id(source_path, start_offset, end_offset, node_kind),
                    node_kind=node_kind,  # type: ignore[arg-type]
                    display_name=display_name,
                    package_name=current_package or package_name_override,
                    enclosing_routines=routine_stack,
                    source_map=source_map,
                    signature_text=signature,
                    conditional_state=conditional_state_for_range(
                        conditional_regions,
                        start_offset=start_offset,
                        end_offset=end_offset,
                    ),
                )
            )

        next_depth = routine_depth + (1 if is_routine else 0)
        next_stack = routine_stack + ((display_name,) if is_routine and display_name else ())
        for child in getattr(context, "children", None) or []:
            if isinstance(child, ParserRuleContext):
                visit(
                    child,
                    current_package or package_name_override,
                    next_depth,
                    next_stack,
                )

    visit(tree, package_name_override, 0, ())
    unique = {(node.node_kind, node.source_map.start_offset, node.source_map.end_offset): node for node in nodes}
    return tuple(sorted(unique.values(), key=lambda node: (node.source_map.start_offset, node.node_kind)))


def _signature_text(text: str, *, is_spec: bool) -> str:
    if is_spec:
        return " ".join(text.strip().split())
    lexer = PlSqlLexer(InputStream(text))
    lexer.removeErrorListeners()
    stream = CommonTokenStream(lexer)
    stream.fill()
    paren_depth = 0
    for token in stream.tokens:
        if token.type == PlSqlLexer.LEFT_PAREN:
            paren_depth += 1
        elif token.type == PlSqlLexer.RIGHT_PAREN:
            paren_depth = max(0, paren_depth - 1)
        elif paren_depth == 0 and token.type in {PlSqlLexer.IS, PlSqlLexer.AS}:
            return " ".join(text[: token.start].strip().split())
    return " ".join(text.strip().split())[:1000]


def _whole_file_map(source_text: str, source_path: str) -> SourceMap:
    line_count = max(1, len(source_text.splitlines()))
    return SourceMap(
        source_path=source_path,
        start_line=1,
        end_line=line_count,
        start_offset=0,
        end_offset=len(source_text),
    )


def _segment_id(source_path: str, start: int, end: int, kind: str) -> str:
    return hashlib.sha256(f"{source_path}|{start}|{end}|{kind}".encode("utf-8")).hexdigest()
