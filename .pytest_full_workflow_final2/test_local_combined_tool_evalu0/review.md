# Bounded agentic tools SME review packet

- Manifest SHA-256: `4444444444444444444444444444444444444444444444444444444444444444`
- Report identity: `27cb0ad6441f835d1a8d6753f8edda774793f0f5d1e67048d6834e1785ca0775`
- Policy SHA-256: `dcdd5b90790c77e913ddcfeaa619715311b8cdca9491d8b78a806472e5b789e2`
- Positive results: **1/1**
- Safety results: **5/5**
- External API calls: **0**
- Review status: **draft**

Review whether each natural question, expected evidence identity, and required
lineage behavior is correct. This packet intentionally omits full FDD/code text.

## 1. tool-combined-test-001

- Question: Explain the AML process integration with FlagRight.
- Mode/tools: `combined` / `fdd_search, code_search, impact_graph`
- Structural result: **pass**
- Expected FDD documents: `['FDD-AML-R24']`
- Expected code paths: `['pkg_aml_custom.sql']`
- Expected code symbols: `['SP_PROCESS_AML']`
- Reviewed lineage required: **True**

### Deterministic checks

- `fdd_documents`: **pass**; expected `['FDD-AML-R24']`; observed `['FDD-AML-R24']`
- `code_paths`: **pass**; expected `['pkg_aml_custom.sql']`; observed `['pkg_aml_custom.sql']`
- `code_symbols`: **pass**; expected `['SP_PROCESS_AML']`; observed `['SP_PROCESS_AML']`
- `reviewed_lineage`: **pass**; expected `['reviewed_edge']`; observed `['reviewed_edge']`

SME verdict: accepted | corrected | needs_more_context
SME corrected expectation:
SME rationale:
Required follow-up:
