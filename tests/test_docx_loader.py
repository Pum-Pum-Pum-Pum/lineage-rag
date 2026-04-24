from pathlib import Path

from app.ingestion.docx_loader import discover_docx_files


def test_discover_docx_files_filters_temp_files(tmp_path: Path) -> None:
    (tmp_path / "FS_FCIS_14.4.0.0.0$ASNB_R2_PNB_Branch Online Reports(BOR)_v1.2.docx").write_text("x")
    (tmp_path / "~$_FCIS_14.7.0.0.0$ASNB_R24_Teller_Branch_Reports_Realignment_v1.0.docx").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    discovered = discover_docx_files(tmp_path)

    assert len(discovered) == 1
    assert discovered[0].file_name == "FS_FCIS_14.4.0.0.0$ASNB_R2_PNB_Branch Online Reports(BOR)_v1.2.docx"
    assert discovered[0].is_temporary is False


def test_discover_docx_files_missing_directory() -> None:
    missing_path = Path("definitely_missing_directory_for_test")

    try:
        discover_docx_files(missing_path)
    except FileNotFoundError as exc:
        assert "Input directory does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing directory")
