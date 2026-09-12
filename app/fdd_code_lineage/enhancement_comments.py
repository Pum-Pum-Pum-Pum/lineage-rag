"""Extract enhancement evidence from PL/SQL comments, never from SQL strings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from antlr4 import InputStream, Token

from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer


IDENTITY = re.compile(
    r"(?P<release>FCIS_[0-9.]+\$[A-Z0-9]+_R[0-9]+)"
    r"\s*(?:\((?P<requirement>REQ[0-9]+)\))?\s*[-–]\s*(?P<title>[^\r\n]+)",
    re.IGNORECASE,
)


def identity_key(release: str, requirement: str | None, title: str) -> str:
    # Deliberately do not strip Part2, SCR, or Sonar Fixes from the title.
    return "|".join((release.upper(), (requirement or "").upper(),
                     " ".join(title.casefold().split())))


@dataclass(frozen=True)
class EnhancementRegion:
    identity: str
    label: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int

    def as_dict(self) -> dict:
        return asdict(self)


def extract_regions(source: str) -> tuple[list[EnhancementRegion], list[dict], list[dict]]:
    """Return paired regions, file-only mentions, and unmatched/crossing diagnostics."""
    lexer = PlSqlLexer(InputStream(source))
    lexer.removeErrorListeners()
    comment_types = {lexer.SINGLE_LINE_COMMENT, lexer.MULTI_LINE_COMMENT, lexer.REMARK_COMMENT}
    stack: list[dict] = []
    regions: list[EnhancementRegion] = []
    mentions: list[dict] = []
    diagnostics: list[dict] = []
    while True:
        token = lexer.nextToken()
        if token.type == Token.EOF:
            break
        if token.type not in comment_types:
            continue
        offset = token.start
        for line_index, line in enumerate(token.text.splitlines(keepends=True)):
            match = IDENTITY.search(line)
            if match:
                tail = match['title'].strip().rstrip('*/').strip()
                parts = [part.strip() for part in tail.split('::')]
                title = parts[0]
                key = identity_key(match['release'], match['requirement'], title)
                boundary = parts[-1].casefold() if len(parts) > 1 else ''
                marker = dict(identity=key, label=match['release'].upper() +
                              (f" ({match['requirement'].upper()})" if match['requirement'] else '') +
                              ' - ' + title, line=token.line + line_index, offset=offset)
                if boundary == 'start':
                    stack.append(marker)
                elif boundary == 'end':
                    if stack and stack[-1]['identity'] == key:
                        start = stack.pop()
                        regions.append(EnhancementRegion(
                            key, start['label'], start['line'], marker['line'],
                            start['offset'], offset + len(line)))
                    else:
                        diagnostics.append(dict(kind='unmatched_or_crossing_end', **marker))
                        # Invalidate open regions at a crossed boundary; never guess pairing.
                        diagnostics.extend(dict(kind='invalidated_start', **entry) for entry in stack)
                        stack.clear()
                else:
                    mentions.append(marker)
            offset += len(line)
    diagnostics.extend(dict(kind='unmatched_start', **entry) for entry in stack)
    return sorted(regions, key=lambda item: item.start_offset), mentions, diagnostics
