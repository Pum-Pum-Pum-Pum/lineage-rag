# Pinned PL/SQL grammar

The grammar is Apache-2.0 source from `antlr/grammars-v4`, pinned to commit:

```text
a7704d4c029c33a89818ac103f758f7c72d8d16c
```

Source hashes:

```text
FE8A00E31E1B7F8C2F26A6143CA49A3122FC7C509013D15357F8A9918DDADB84 PlSqlLexer.g4
C4B6B49EFD217CF6FAA770F607CFB2F88830FF09384E8E5F19FBA189B74385E4 PlSqlParser.g4
A078FCACEF0A3D300A492988377DEFD788BC5287B67BAE4F25CAD53C9AF4D727 PlSqlLexerBase.py
BDD1F998AEF1127B98D7CCD950FFA48472D04C1F44F36DD5DB44E7E73413BD86 PlSqlParserBase.py
```

Generate with ANTLR `4.13.2` and the matching Python runtime:

```powershell
java -jar antlr-4.13.2-complete.jar `
  -Dlanguage=Python3 -visitor -no-listener -Xexact-output-dir `
  -o ../../generated/plsql PlSqlLexer.g4 PlSqlParser.g4
```

The generator jar SHA-256 used for this generation was:

```text
EAE2DFA119A64327444672AFF63E9EC35A20180DC5B8090B7A6AB85125DF4D76
```

Copy the two Python base classes beside the generated parser. The generated
copy of `PlSqlParserBase.py` uses package-relative lexer imports so installed
runtime packages do not depend on the current working directory.

The upstream grammar emits target-neutral `this.<predicate>(...)` references
in both generated Python files. Replace `this.` with `self.` in the generated
lexer and parser. This is a documented Python-target compatibility fix; the
vendored upstream grammar remains byte-identical to its recorded hash.
