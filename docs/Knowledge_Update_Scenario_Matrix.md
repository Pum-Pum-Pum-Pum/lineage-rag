# Knowledge Update Scenario Matrix

This numbered matrix is the maintained regression contract for
`run_knowledge_update`. Each new unsupported case must be added here and to a
deterministic test before it becomes an operationally supported path.

| ID | Situation | Expected result |
|---:|---|---|
| 01 | Identical FDD input | No-op; no vector call or promotion |
| 02 | Additive FDD-only folder | Baseline documents retained |
| 03 | Complete code-only snapshot | Missing old file reported as deletion |
| 04 | New FDD and code together | One compatible final release |
| 05 | Same FDD path, new bytes | Explicit replacement manifest required |
| 06 | Full-replacement FDD revision | Old source superseded, not removed twice |
| 07 | Supplementary FDD revision | Both sources retained |
| 08 | Withdrawal | Reviewed retirement; no silent disappearance |
| 09 | Same hash under a new path | Move only when unambiguous |
| 10 | Case-insensitive path collision | Stop before copying |
| 11 | Symlink/junction escape | Stop before copying |
| 12 | Embedded workbook-only change | New FDD units are discovered and evaluated |
| 13 | Missing/unparseable attachment | Stop before embedding |
| 14 | New FDD, unchanged code | Discover candidate without code re-embedding |
| 15 | New code, old FDD | Discover candidate against retained FDD corpus |
| 16 | Changed caller/validation context | Invalidate affected carry-forward |
| 17 | Repeated routine/overload | Require exact selector evidence |
| 18 | Table-heavy routine | Return code operation evidence without DB invention |
| 19 | No approved FDD link | Code evidence; visibly unreviewed boundary |
| 20 | New evidence conflicts with accepted link | Review finding; never automatic replacement |
| 21 | Previously deferred candidate gains evidence | New review item only |
| 22 | Legacy document-level lineage | Carry only with unchanged document/context |
| 23 | Cache compatible unchanged input | Zero unexpected cache misses |
| 24 | Model/chunker/parser change | Explicit migration; no cross-space reuse |
| 25 | Budget/quota exhaustion | Stop with receipts and no automatic retry |
| 26 | Unknown request outcome | Reconciliation required; never resend |
| 27 | Stale/corrupted review or artifact | Stop with exact diagnostic |
| 28 | Qdrant store lock/collection collision | Stop; never delete or overwrite |
| 29 | Concurrent promotion or changed runtime | New request required |
| 30 | Restart mismatch/rollback | No completion; restore prior compatible state with code disabled |
