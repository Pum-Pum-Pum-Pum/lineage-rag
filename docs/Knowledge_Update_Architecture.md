# Knowledge Update Architecture

The system serves one compatible release combination, not independent FDD and
code versions.

```text
Immutable FDD evidence ── extraction/chunking ── FDD vectors + lexical units ┐
                                                                            ├─ reviewed lineage ─ MCP retrieval
Complete code snapshot ─ parse/dependencies ── code vectors + analysis ────┘
                         \_________________ cache reuse _________________/
```

FDDs describe intended behaviour; parsed source is implementation evidence.
Neither is silently treated as authoritative over the other. A reviewed
lineage record classifies baseline requirement, enhancement, supporting
behaviour, conflict, or no reviewed documentation link. When they conflict,
responses present the implemented code evidence and label the documentation
conflict; they do not invent reconciliation.

## Artifact levels

1. A source manifest records files, content hashes, logical identities,
   supersession and controlled copies.
2. Prepared artifacts freeze extraction/parser contracts and cache keys.
3. Embedded artifacts add vectors; exact text/model/version compatibility is
   required before vector reuse.
4. Candidate lineage is local discovery evidence only.
5. Reviewed lineage binds an SME decision to passages, code targets and
   relevant caller/dependency context.
6. A release manifest binds the final FDD generation, code snapshot,
   collections, lineage, deferred backlog, evaluations and runtime files.

This separation is why a new FDD can be compared with unchanged code without
re-embedding code, and a changed source file can re-use unchanged chunks.

## Automated work versus human decisions

Automated processing copies inputs, hashes, parses, detects change impact,
reuses compatible vectors, performs full local discovery and runs evaluation.
Humans approve paid inputs, decide uncertain/invalidated mapping findings, and
approve an exact final release hash. A metadata-only correction may reuse
vectors. A correction that changes embedding text freezes a new additional
input request.

## Serving and recovery

Activation first verifies staged artifacts and store ownership, copies a new
FDD lexical directory without overwriting prior indexes, then atomically
switches all FDD and code settings. MCP startup publishes a signed receipt
only when the actual restarted process serves the requested combination. The
receipt, rather than `.env` alone, completes a release.

The prior compatible selection is retained for rollback. A code-disabled
rollback is intentional: it avoids claiming a previously mixed or stale
combined runtime is safe.
