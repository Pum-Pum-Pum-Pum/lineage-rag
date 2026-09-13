# Bounded-tool formal manual UAT review packet

- Manifest SHA-256: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
- Batch identity: `4db9a4af92148af3e49f2ebfe65a692a2873304d0061b7da94edb54950ed5741`
- Diagnostic results: **1/1**
- Review status: **draft**
- External API calls: **0**

Review the local report named for each case. Those reports contain internal
source text; this packet contains identities and checks only.

## 1. uat-combined-test-001

- Source reviewed case: `reviewed-source-001`
- Question: Explain the AML transaction integration and visible custom code.
- Mode: `combined`
- Expected outcome: `evidence`
- Diagnostic result: **pass**
- Local report: `C:\OC\AI\AI_Projects\Culling Blade Lineage- Gen AI RAG System\.pytest_r3_full_suite\test_manual_uat_batch_is_draft0\case.json`
- Local report SHA-256: `f7df4c846e7d48310d8f6fc6d575c54f83bcf1687314768e4aabacf799349904`

### Checks

- `fdd_documents`: **pass**; expected `['FDD-AML-R24']`; observed `['FDD-AML-R24']`
- `code_paths`: **pass**; expected `['pkg_aml_custom.sql']`; observed `['pkg_aml_custom.sql']`
- `code_symbols`: **pass**; expected `['SP_PROCESS_AML']`; observed `['SP_PROCESS_AML']`
- `reviewed_lineage`: **pass**; expected `['reviewed_edge']`; observed `['reviewed_edge']`
- `qualified_unknown`: **pass**; expected `[]`; observed `[]`

SME verdict: accepted | retrieval_gap | wrong_source | needs_more_context
SME rationale:
Required follow-up:
