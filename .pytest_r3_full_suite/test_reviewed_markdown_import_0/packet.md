# Custom-code dependency SME review packet

- Snapshot: `fci-custom-r1-abc`
- Parser generation: `plsql_antlr_4_13_2_analysis_v12`
- Policy SHA-256: `dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd`
- Packet SHA-256: `6274fd5424e4e6052a67d8632ff1a91ce1a2132050a986420864d667042fd72a`
- Review cases: 1
- Source occurrences represented: 1
- Review status: `draft`

For each case, confirm or correct the dependency kind and resolution state.
Do not infer runtime behavior beyond the displayed static evidence.

## 1. PKG_MISSING_CUSTOM.GET_VALUE

- Review ID: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`
- Proposed kind: `routine_call`
- Proposed state: `custom_source_missing`
- Confidence: `high`
- Reason: Custom source is absent.
- Occurrences: 1

### Evidence: `pkg_report_custom.sql:10`

```sql
000010: pkg_missing_custom.get_value();
```

SME verdict: accepted
SME corrected kind/state: 
SME rationale: Source will be supplied later.
