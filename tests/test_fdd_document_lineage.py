from pathlib import Path

from app.ingestion.docx_loader import DiscoveredDocxFile
from app.ingestion.fdd_document_lineage import (
    FddDocumentLineagePolicy,
    select_current_document_sources,
)


def _source(tmp_path: Path, name: str) -> DiscoveredDocxFile:
    path = tmp_path / name
    path.write_bytes(b"fixture")
    return DiscoveredDocxFile(file_name=name, file_path=path, is_temporary=False)


def _policy(*keys: str) -> FddDocumentLineagePolicy:
    return FddDocumentLineagePolicy(
        schema_version="test",
        full_replacement_keys=frozenset(key.casefold() for key in keys),
        sha256="policy-hash",
    )


def test_full_replacement_stream_selects_latest_numeric_document_revision(tmp_path: Path) -> None:
    v229 = _source(tmp_path, "FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services_v2.29.docx")
    v231 = _source(tmp_path, "FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services_v2.31.docx")
    selection = select_current_document_sources(
        [v231, v229],
        policy=_policy("FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services"),
    )

    assert selection.current_sources == (v231,)
    assert selection.superseded_sources == (v229,)
    assert selection.policy_sha256 == "policy-hash"


def test_unconfigured_revisioned_stream_remains_independent_evidence(tmp_path: Path) -> None:
    v10 = _source(tmp_path, "FS_FCIS_14.4.0.0.0$ASNB_R4_Unreviewed Stream_v1.0.docx")
    v11 = _source(tmp_path, "FS_FCIS_14.4.0.0.0$ASNB_R4_Unreviewed Stream_v1.1.docx")
    selection = select_current_document_sources([v10, v11], policy=_policy())

    assert selection.current_sources == (v10, v11)
    assert selection.superseded_sources == ()
