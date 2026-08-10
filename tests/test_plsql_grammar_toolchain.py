from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path


GRAMMAR_ROOT = Path("app/code_ingestion/grammar/plsql")
EXPECTED_HASHES = {
    "PlSqlLexer.g4": "fe8a00e31e1b7f8c2f26a6143ca49a3122fc7c509013d15357f8a9918ddadb84",
    "PlSqlParser.g4": "c4b6b49efd217cf6faa770f607cfb2f88830ff09384e8e5f19fba189b74385e4",
    "PlSqlLexerBase.py": "a078fcacef0a3d300a492988377defd788bc5287b67bae4f25cad53c9af4d727",
    "PlSqlParserBase.py": "bdd1f998aef1127b98d7ccd950ffa48472d04c1f44f36dd5db44e7e73413bd86",
}


def test_antlr_runtime_and_vendored_grammar_are_pinned() -> None:
    assert version("antlr4-python3-runtime") == "4.13.2"
    for filename, expected_hash in EXPECTED_HASHES.items():
        assert hashlib.sha256((GRAMMAR_ROOT / filename).read_bytes()).hexdigest() == expected_hash


def test_generated_python_has_documented_target_fix_and_no_machine_path() -> None:
    generated_root = Path("app/code_ingestion/generated/plsql")
    generated = [
        generated_root / "PlSqlLexer.py",
        generated_root / "PlSqlParser.py",
    ]

    for path in generated:
        text = path.read_text(encoding="utf-8")
        assert "this." not in text
        assert "C:\\" not in text
        assert "C:/" not in text
        assert "ANTLR 4.13.2" in text[:200]
